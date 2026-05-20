#!/usr/bin/env bash
# Pod runbook: variant-aware eval across the cumulative chain, sharded
# across all available GPUs.
#
# The default eval-at-end loop in train_chain_extension.sh runs ONE eval
# process at a time on ONE GPU, so 3 of 4 H100s sit idle. This script
# instead launches one eval process per GPU, each owning a contiguous
# chunk of years via CUDA_VISIBLE_DEVICES pinning. Combined with batched
# score_continuations in talkie.py (which makes each eval ~5x faster on
# its own), the full 90-year eval drops from ~19 hr to ~1-1.5 hr on
# 4x H100.
#
# Usage:
#   bash scripts/pod/run_variants_eval_sharded.sh              # all years 1931-2020
#   bash scripts/pod/run_variants_eval_sharded.sh 1931 2020    # explicit range
#   bash scripts/pod/run_variants_eval_sharded.sh 1950 1999    # subrange
#
# Output: same per-year eval.json files as serial eval — fully compatible
# with downstream plotting scripts. Each shard's logs go to
# /tmp/eval_shard_<N>.log; merged tail can be followed via the printed hint.
#
# Prerequisites:
#   - All N year-models in the requested range have trained checkpoints
#     at $POLICY_PRED_DATA_ROOT/experiments/policy_<Y>_naive/checkpoint
#   - data_artifacts/question_variants/ extracted with 211 *.jsonl files
#   - repo at a commit containing the batched score_continuations
#     (talkie.py) AND the lookback filter (evaluators/policy_battery_variants.py)
#
set -uo pipefail   # NOT -e — let individual evals fail without killing the wave

START_YEAR="${1:-1931}"
END_YEAR="${2:-2020}"

# GPU count: auto-detect; override with NUM_GPUS=N env var.
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [[ -z "$NUM_GPUS" || "$NUM_GPUS" -lt 1 ]]; then NUM_GPUS=1; fi

export TALKIE_WEIGHTS_DIR="${TALKIE_WEIGHTS_DIR:-${HOME}/talkie_base}"
export POLICY_PRED_DATA_ROOT="${POLICY_PRED_DATA_ROOT:-${HOME}/policy_pred_data}"

echo "============================================================="
echo "  Sharded variant-aware eval"
echo "============================================================="
echo "  Year range:    ${START_YEAR}..${END_YEAR}"
echo "  GPU shards:    ${NUM_GPUS}"
echo "  Logs:          /tmp/eval_shard_<N>.log"
echo

# --- Activate venv if present ---
if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# --- Build the year list, skipping years whose checkpoint is missing ---
years=()
for y in $(seq "$START_YEAR" "$END_YEAR"); do
    ckpt="${POLICY_PRED_DATA_ROOT}/experiments/policy_${y}_naive/checkpoint"
    if [[ -f "$ckpt/adapter_config.json" ]]; then
        years+=("$y")
    else
        echo "  SKIP year=$y (no adapter at $ckpt)"
    fi
done
n_total=${#years[@]}
if [[ "$n_total" -eq 0 ]]; then
    echo "Nothing to evaluate."
    exit 1
fi
echo "  Years to eval: $n_total"
echo

# --- Build shards by interleaving (round-robin) so each shard has a mix
# of early and late years. Even sharding would put all the heavy late
# years on one shard.
declare -A shard_years
for i in "${!years[@]}"; do
    shard=$(( i % NUM_GPUS ))
    shard_years[$shard]+="${years[$i]} "
done

# --- Launch one background process per shard, pinned to its GPU ---
start_time=$(date +%s)
pids=()
for shard in $(seq 0 $((NUM_GPUS - 1))); do
    yrs="${shard_years[$shard]:-}"
    [[ -z "$yrs" ]] && continue
    log="/tmp/eval_shard_${shard}.log"
    echo "  shard=$shard  GPU=$shard  years: $yrs  log: $log"
    (
        export CUDA_VISIBLE_DEVICES=$shard
        for y in $yrs; do
            yr_start=$(date +%s)
            python3 eval.py \
                --experiment "experiments/policy_${y}_naive.yaml" \
                --evaluator policy_battery_variants \
                --force \
                > "${log}.${y}" 2>&1 \
                && status="OK" || status="FAIL"
            yr_elapsed=$(( $(date +%s) - yr_start ))
            echo "[shard=$shard] year=$y $status (${yr_elapsed}s)" | tee -a "$log"
        done
    ) &
    pids+=($!)
done

echo
echo "Launched ${#pids[@]} shards. Follow live progress with:"
echo "  tail -f /tmp/eval_shard_*.log"
echo "Or inspect per-year logs at /tmp/eval_shard_<S>.log.<YYYY>"
echo

# --- Wait for all shards ---
n_failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        n_failed=$((n_failed + 1))
    fi
done

total_elapsed=$(( $(date +%s) - start_time ))
n_eval_json=$(find "${POLICY_PRED_DATA_ROOT}/experiments" -name eval.json \
    -newer "/tmp/eval_shard_0.log" 2>/dev/null | wc -l)

echo
echo "============================================================="
echo "  Sharded eval complete"
echo "============================================================="
echo "  Total wall time:  $((total_elapsed / 3600))h $((total_elapsed % 3600 / 60))m $((total_elapsed % 60))s"
echo "  Years requested:  $n_total"
echo "  eval.json written: $n_eval_json"
echo "  Failed shards:    $n_failed"
echo
echo "Pull results back to local with:"
echo "  cd $POLICY_PRED_DATA_ROOT/experiments"
echo "  tar czf /tmp/chain_evals.tgz policy_*/eval.json"
echo "  # WinSCP /tmp/chain_evals.tgz to local C:/tmp/"
