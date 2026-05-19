#!/usr/bin/env bash
# Pod runbook: cumulative LoRA chain training for years 1936-2020.
#
# Each year-Y model = base + cumulative LoRA(1931..Y), trained with
# train.py --init-from <prior-year-checkpoint> --data <Y-synth>.
#
# Prerequisites:
#   1. ~/policy_synth_1936_2020.zip uploaded via WinSCP (~700 MB)
#   2. Existing 1931_naive .. 1935_naive checkpoints already on pod
#      under $POLICY_PRED_DATA_ROOT/experiments/ (from earlier 1k-seed run)
#   3. Repo on commit ad3036b or later
#
# Usage:
#   bash scripts/pod/train_chain_extension.sh            # train 1936-2020 (default)
#   bash scripts/pod/train_chain_extension.sh 1936 1955  # train a sub-range
#   bash scripts/pod/train_chain_extension.sh 1936 2020 eval_every_year  # eval at each step (slow)
#
# Behavior:
#   - Idempotent: skips years whose checkpoint already exists
#   - Each year init-froms the prior year (cumulative chain)
#   - By default, eval only at the END to save time. Pass "eval_every_year"
#     as the 3rd arg to eval after each year's training.
#
# Wall time at 5k seeds (training scales ~linearly with record count vs 1k):
#   ~35-45 min training per year on 4-GPU pod
#   ~5-10 min variant eval per year
#   Train-only, 1936-2020 (85 yrs): ~50-65 hr
#   Train+eval each year:           ~60-80 hr (~+15h for evals)
#
set -uo pipefail   # NOT -e — continue past per-year failures so the chain can recover

START_YEAR="${1:-1931}"
END_YEAR="${2:-2020}"
EVAL_MODE="${3:-eval_at_end}"   # "eval_every_year" | "eval_at_end" | "no_eval"
# GPU count: override with `export NUM_PROCESSES=8` (or pass via env); auto-detect otherwise.
NUM_PROCESSES="${NUM_PROCESSES:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [[ -z "$NUM_PROCESSES" || "$NUM_PROCESSES" -lt 1 ]]; then NUM_PROCESSES=4; fi

# Resolve env vars.
export TALKIE_WEIGHTS_DIR="${TALKIE_WEIGHTS_DIR:-${HOME}/talkie_base}"
export POLICY_PRED_DATA_ROOT="${POLICY_PRED_DATA_ROOT:-${HOME}/policy_pred_data}"

DATA_DIR="${HOME}/data"
EXPS_DIR="${POLICY_PRED_DATA_ROOT}/experiments"

echo "============================================================="
echo "  Pod runbook: cumulative chain training 1936-2020"
echo "============================================================="
echo "  Year range:    ${START_YEAR}..${END_YEAR}"
echo "  Eval mode:     ${EVAL_MODE}"
echo "  GPUs:          ${NUM_PROCESSES}"
echo "  Talkie:        $TALKIE_WEIGHTS_DIR"
echo "  Data root:     $POLICY_PRED_DATA_ROOT"
echo "  Synth files:   $DATA_DIR/policy_<Y>_naive.jsonl"
echo

# --- Sanity checks ---
if [[ ! -d "$TALKIE_WEIGHTS_DIR" ]] || [[ ! -f "$TALKIE_WEIGHTS_DIR/final.ckpt" ]]; then
    echo "ERROR: Talkie weights missing at $TALKIE_WEIGHTS_DIR"
    exit 1
fi

# Make sure venv is active.
if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# --- Verify prior chain link is present (or special-case year 1931 = first link) ---
if [[ "$START_YEAR" -eq 1931 ]]; then
    echo "  Year 1931 is the first link in the chain → no --init-from (trains fresh on base)."
else
    prev_ckpt="${EXPS_DIR}/policy_$((START_YEAR - 1))_naive/checkpoint"
    if [[ ! -d "$prev_ckpt" ]]; then
        echo "ERROR: previous-year checkpoint not found at $prev_ckpt"
        echo "       Cumulative chain needs the prior year's adapter to --init-from."
        echo "       Re-train earlier years first, or start at START_YEAR=1931."
        exit 1
    fi
    echo "  Initial init-from: $prev_ckpt (year $((START_YEAR - 1)))"
fi
echo

# --- Loop years ---
start_time=$(date +%s)
n_trained=0
n_skipped=0
n_failed=0

for y in $(seq "$START_YEAR" "$END_YEAR"); do
    yr_dir="${EXPS_DIR}/policy_${y}_naive"
    ckpt_dir="${yr_dir}/checkpoint"
    synth_file="${DATA_DIR}/policy_${y}_naive.jsonl"
    prev_year=$((y - 1))
    prev_ckpt="${EXPS_DIR}/policy_${prev_year}_naive/checkpoint"

    echo "============================================================"
    echo "  YEAR $y  (init-from $prev_year)"
    echo "============================================================"

    # Only skip if the checkpoint actually contains an adapter — a bare empty
    # directory left over from a crashed earlier run is NOT a valid checkpoint
    # and would fool a subsequent year's --init-from into loading nothing.
    if [[ -f "$ckpt_dir/adapter_config.json" ]]; then
        echo "  SKIP: checkpoint already exists at $ckpt_dir"
        n_skipped=$((n_skipped + 1))
        continue
    fi
    if [[ -d "$ckpt_dir" ]]; then
        echo "  WARN: stub checkpoint dir at $ckpt_dir (no adapter_config.json); removing and retraining"
        rm -rf "$ckpt_dir"
    fi

    if [[ ! -f "$synth_file" ]]; then
        echo "  ERROR: synth file missing at $synth_file"
        n_failed=$((n_failed + 1))
        continue
    fi
    # Year 1931 starts fresh on base — no prior checkpoint expected.
    if [[ "$y" -ne 1931 ]] && [[ ! -d "$prev_ckpt" ]]; then
        echo "  ERROR: prior-year checkpoint missing at $prev_ckpt"
        n_failed=$((n_failed + 1))
        continue
    fi

    n_records=$(wc -l < "$synth_file")
    echo "  Training on $synth_file ($n_records records)"

    # Year 1931 is the first link → no --init-from. Every other year init-froms the prior.
    if [[ "$y" -eq 1931 ]]; then
        init_from_args=""
        echo "  (no --init-from; year 1931 trains fresh on base)"
    else
        init_from_args="--init-from $prev_ckpt"
        echo "  --init-from $prev_ckpt"
    fi

    yr_start=$(date +%s)
    if accelerate launch --num_processes=$NUM_PROCESSES --mixed_precision=no train.py \
        --experiment "experiments/policy_${y}_naive.yaml" \
        --data "$synth_file" \
        --batch-size 2 \
        $init_from_args
    then
        yr_elapsed=$(( $(date +%s) - yr_start ))
        echo "  TRAIN DONE: $((yr_elapsed / 60)) min"
        n_trained=$((n_trained + 1))
    else
        echo "  TRAIN FAILED for year $y. Stopping (cumulative chain can't skip a year)."
        n_failed=$((n_failed + 1))
        break
    fi

    # Optional per-year eval.
    if [[ "$EVAL_MODE" == "eval_every_year" ]]; then
        echo "  --- Eval $y (variant-aware) ---"
        python3 eval.py \
            --experiment "experiments/policy_${y}_naive.yaml" \
            --evaluator policy_battery_variants \
            --force || true
    fi
done

total_elapsed=$(( $(date +%s) - start_time ))

# Final eval if not done per-year.
if [[ "$EVAL_MODE" == "eval_at_end" ]] && [[ $n_trained -gt 0 ]]; then
    echo
    echo "============================================================="
    echo "  Final variant-aware eval for ALL trained years"
    echo "============================================================="
    for y in $(seq "$START_YEAR" "$END_YEAR"); do
        ckpt_dir="${EXPS_DIR}/policy_${y}_naive/checkpoint"
        if [[ -d "$ckpt_dir" ]]; then
            echo "  Eval $y..."
            python3 eval.py \
                --experiment "experiments/policy_${y}_naive.yaml" \
                --evaluator policy_battery_variants \
                --force || true
        fi
    done
fi

echo
echo "============================================================="
echo "  Campaign complete"
echo "============================================================="
echo "  Total wall time:  $((total_elapsed / 3600))h $((total_elapsed % 3600 / 60))m"
echo "  Years trained:    $n_trained"
echo "  Years skipped:    $n_skipped"
echo "  Years failed:     $n_failed"
echo
echo "Pull results back to local with:"
echo "  cd $POLICY_PRED_DATA_ROOT/experiments"
echo "  tar czf /tmp/chain_evals.tgz policy_19*/eval.json policy_20*/eval.json policy_1931_base/eval.json"
echo "  # WinSCP /tmp/chain_evals.tgz to local C:/tmp/"
