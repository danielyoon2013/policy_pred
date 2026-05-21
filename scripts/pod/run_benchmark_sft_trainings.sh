#!/usr/bin/env bash
# Pod runbook: train + eval the 4 benchmark LoRA-SFT cells.
#
# This is Phase 3 step 2 — the methodological smoke test. Each cell:
#   1. accelerate launch ... train.py ... --sft   (LoRA-SFT adapter)
#   2. python3 eval.py ... --force                 (eval trained adapter)
#
# Pairs with run_benchmark_base_evals.sh (already done) to give the
# 8-number comparison table at the end:
#
#                      | ARC   | GSM8K |
#   Talkie base        |  ??   |  ??   |   <- from base_evals
#   Talkie + LoRA-SFT  |  ??   |  ??   |   <- from THIS script
#   Nanochat base      |  ??   |  ??   |   <- from base_evals
#   Nanochat + LoRA-SFT|  ??   |  ??   |   <- from THIS script
#
# Cells trained sequentially because each uses all 4 GPUs via DDP.
#
# Idempotent: skips any cell whose adapter checkpoint already exists,
# and any cell whose backend weights are missing. Useful for retries
# after partial completion.
#
# Usage:
#   bash scripts/pod/run_benchmark_sft_trainings.sh
#
# Wall time on 4x H100:
#   Talkie 13B SFT  (~2000 steps): ~60-80 min per cell
#   Nanochat 1.36B SFT (same data): ~10-15 min per cell
#   Total:                          ~2.5-3 hr
#
# Logs land in /tmp/benchmark_sft_<cell>.log per cell.

set -uo pipefail

export TALKIE_WEIGHTS_DIR="${TALKIE_WEIGHTS_DIR:-${HOME}/talkie_base}"
export NANOCHAT_WEIGHTS_DIR="${NANOCHAT_WEIGHTS_DIR:-${HOME}/nanochat_base}"
export POLICY_PRED_DATA_ROOT="${POLICY_PRED_DATA_ROOT:-${HOME}/policy_pred_data}"

# GPU count for accelerate launch. Auto-detect, override via NUM_PROCESSES env.
NUM_PROCESSES="${NUM_PROCESSES:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [[ -z "$NUM_PROCESSES" || "$NUM_PROCESSES" -lt 1 ]]; then NUM_PROCESSES=4; fi

echo "============================================================="
echo "  Pod runbook: benchmark LoRA-SFT training (4 cells)"
echo "============================================================="
echo "  Talkie weights:    $TALKIE_WEIGHTS_DIR"
echo "  Nanochat weights:  $NANOCHAT_WEIGHTS_DIR"
echo "  Data root:         $POLICY_PRED_DATA_ROOT"
echo "  GPUs:              $NUM_PROCESSES"
echo

# --- venv ---
if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# --- Backend availability ---
HAS_TALKIE=1
if [[ ! -f "$TALKIE_WEIGHTS_DIR/final.ckpt" ]]; then
    echo "WARN: Talkie weights not at $TALKIE_WEIGHTS_DIR/final.ckpt -- Talkie cells will be skipped"
    HAS_TALKIE=0
fi
HAS_NANOCHAT=1
if ! ls "$NANOCHAT_WEIGHTS_DIR/base_checkpoints"/d* >/dev/null 2>&1; then
    echo "WARN: Nanochat weights not under $NANOCHAT_WEIGHTS_DIR/base_checkpoints/d* -- nanochat cells will be skipped"
    HAS_NANOCHAT=0
fi

# --- Training data symlinks ---
for f in data_links/benchmarks/arc_train.jsonl data_links/benchmarks/gsm8k_train.jsonl; do
    if [[ ! -e "$f" ]] && [[ ! -L "$f" ]]; then
        echo "ERROR: missing $f"
        echo "  Create a symlink to wherever hist_llm's gen_a / gen_c jsonl lives on this pod."
        exit 1
    fi
done

# --- Cell definitions: (yaml_stem, backend, expected_min_wall_sec) ---
# Sequential order: nanochat cells first (faster, fail-fast on bugs),
# then talkie cells (longer wall time, last to burn).
declare -a CELLS=(
    "nanochat_arc   nanochat"
    "nanochat_gsm8k nanochat"
    "talkie_arc     talkie"
    "talkie_gsm8k   talkie"
)

# --- Loop ---
start_time=$(date +%s)
n_trained=0
n_skipped=0
n_failed=0

train_and_eval() {
    local stem="$1"
    local backend="$2"
    local yaml="experiments/benchmark/${stem}.yaml"
    local exp_name="benchmark_${stem}"
    local ckpt_dir="${POLICY_PRED_DATA_ROOT}/experiments/${exp_name}/checkpoint"
    local log="/tmp/benchmark_sft_${stem}.log"

    echo
    echo "============================================================"
    echo "  CELL: ${stem}  (backend=${backend})"
    echo "============================================================"

    # Backend availability skip
    if [[ "$backend" == "talkie" ]] && [[ "$HAS_TALKIE" -eq 0 ]]; then
        echo "  SKIP: Talkie weights unavailable"
        n_skipped=$((n_skipped + 1))
        return
    fi
    if [[ "$backend" == "nanochat" ]] && [[ "$HAS_NANOCHAT" -eq 0 ]]; then
        echo "  SKIP: Nanochat weights unavailable"
        n_skipped=$((n_skipped + 1))
        return
    fi

    # Idempotency: skip if adapter already trained (and not partial)
    if [[ -f "${ckpt_dir}/adapter_config.json" ]]; then
        echo "  SKIP TRAIN: adapter already exists at $ckpt_dir"
        # but still re-eval, force=true, to ensure eval.json reflects this run
    else
        if [[ -d "$ckpt_dir" ]]; then
            echo "  WARN: stub checkpoint dir present (no adapter_config.json); removing"
            rm -rf "$ckpt_dir"
        fi

        echo "  --- TRAIN ---"
        local yr_start
        yr_start=$(date +%s)
        if accelerate launch --num_processes="$NUM_PROCESSES" --mixed_precision=no \
            train.py \
                --experiment "$yaml" \
                --batch-size 2 \
                --sft \
            2>&1 | tee "$log"; then
            local yr_elapsed=$(( $(date +%s) - yr_start ))
            echo "  TRAIN DONE: ${stem} in $((yr_elapsed / 60))m $((yr_elapsed % 60))s"
            n_trained=$((n_trained + 1))
        else
            echo "  TRAIN FAILED for ${stem}; see $log"
            n_failed=$((n_failed + 1))
            return
        fi
    fi

    echo "  --- EVAL ---"
    if CUDA_VISIBLE_DEVICES=0 python3 eval.py \
            --experiment "$yaml" \
            --force \
        2>&1 | tee -a "$log"; then
        echo "  EVAL DONE: ${stem}"
    else
        echo "  EVAL FAILED for ${stem}"
        n_failed=$((n_failed + 1))
    fi
}

for entry in "${CELLS[@]}"; do
    # shellcheck disable=SC2086
    read -r stem backend <<< "$entry"
    train_and_eval "$stem" "$backend"
done

total_elapsed=$(( $(date +%s) - start_time ))

echo
echo "============================================================="
echo "  Benchmark SFT matrix complete"
echo "============================================================="
echo "  Wall time:        $((total_elapsed / 3600))h $((total_elapsed % 3600 / 60))m $((total_elapsed % 60))s"
echo "  Cells trained:    $n_trained"
echo "  Cells skipped:    $n_skipped"
echo "  Cells failed:     $n_failed"
echo

# --- Final 8-number comparison table ---
echo "Results table — accuracy on each benchmark:"
echo
printf "  %-35s %-15s\n" "Cell" "Accuracy (n_correct/n_total)"
printf "  %-35s %-15s\n" "-----------------------------------" "------------------------------"
for name in talkie_arc_base talkie_arc talkie_gsm8k_base talkie_gsm8k \
            nanochat_arc_base nanochat_arc nanochat_gsm8k_base nanochat_gsm8k; do
    f="${POLICY_PRED_DATA_ROOT}/experiments/benchmark_${name}/eval.json"
    if [[ -f "$f" ]]; then
        line=$(python3 -c "
import json
with open('$f') as fp:
    d = json.load(fp)
r = d.get('results', {})
acc = r.get('pass_at_1', r.get('accuracy', None))
nc  = r.get('n_correct', '?')
nt  = r.get('n_total',   '?')
if acc is None:
    print('PARSE_ERR')
else:
    print(f'{acc:.4f} ({nc}/{nt})')
" 2>/dev/null || echo "PARSE_ERR")
        printf "  %-35s %s\n" "$name" "$line"
    else
        printf "  %-35s %s\n" "$name" "MISSING"
    fi
done

echo
echo "Pull results back to local with:"
echo "  cd $POLICY_PRED_DATA_ROOT/experiments"
echo "  tar czf /tmp/benchmark_sft_evals.tgz benchmark_*/eval.json"
echo "  # WinSCP /tmp/benchmark_sft_evals.tgz to local C:/tmp/"
