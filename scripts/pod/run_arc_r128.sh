#!/usr/bin/env bash
# Pod runbook: capacity experiment — LoRA r=128 with CoT+MC4 mix on ARC.
#
# Trains nanochat + ARC + r=128 first (fast, ~25 min), then Talkie + ARC +
# r=128 (~70 min). Both use the same CoT+MC4 mixed training data and the
# same MC log-prob eval. Compares results against the r=32 baselines:
#
#                                    base    r=32 mix    r=128 mix (this run)
#   talkie_arc                       0.2622   0.2631      ???
#   nanochat_arc                     0.2631   0.2622      ???
#
# If r=128 lifts ARC meaningfully (>0.28), capacity was the bottleneck.
# If it stays flat, format-match (drop CoT) is the real fix.
#
# Prereqs on pod:
#   - feat/matrix-experiments pulled
#   - data_links/benchmarks/arc_train.jsonl is the CoT+MC4 MIX (138K records)
#     NOT the MC4-only version. To confirm:
#       wc -l data_links/benchmarks/arc_train.jsonl  # expect ~138488
#     If only 69K, re-upload benchmark_training_mix.zip
#   - Talkie + nanochat weights already in place
#
# Usage:
#   bash scripts/pod/run_arc_r128.sh
set -uo pipefail

export TALKIE_WEIGHTS_DIR="${TALKIE_WEIGHTS_DIR:-${HOME}/talkie_base}"
export NANOCHAT_WEIGHTS_DIR="${NANOCHAT_WEIGHTS_DIR:-${HOME}/nanochat_base}"
export POLICY_PRED_DATA_ROOT="${POLICY_PRED_DATA_ROOT:-${HOME}/policy_pred_data}"

NUM_PROCESSES="${NUM_PROCESSES:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [[ -z "$NUM_PROCESSES" || "$NUM_PROCESSES" -lt 1 ]]; then NUM_PROCESSES=4; fi

echo "============================================================="
echo "  Pod runbook: ARC LoRA r=128 capacity experiment (2 cells)"
echo "============================================================="
echo "  GPUs:              $NUM_PROCESSES"
echo "  Data root:         $POLICY_PRED_DATA_ROOT"
echo

# --- venv ---
if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# --- Data sanity ---
if [[ ! -f "data_links/benchmarks/arc_train.jsonl" ]]; then
    echo "ERROR: data_links/benchmarks/arc_train.jsonl missing"; exit 1
fi
n=$(wc -l < data_links/benchmarks/arc_train.jsonl)
echo "  arc_train.jsonl: $n records"
if [[ "$n" -lt 100000 ]]; then
    echo "  WARN: arc_train.jsonl has <100K records — looks like MC4-only,"
    echo "        not the CoT+MC4 mix this experiment expects."
    echo "        Re-upload C:/tmp/benchmark_training_mix.zip and extract to"
    echo "        data_links/benchmarks/ before running this."
    echo "        Continuing anyway in case you intended this."
fi
echo

run_cell() {
    local stem="$1"
    local backend="$2"
    local yaml="experiments/benchmark/${stem}.yaml"
    local exp_name="benchmark_${stem}"
    local ckpt_dir="${POLICY_PRED_DATA_ROOT}/experiments/${exp_name}/checkpoint"
    local log="/tmp/${exp_name}.log"

    echo
    echo "============================================================"
    echo "  CELL: ${stem}  (backend=${backend}, rank=128)"
    echo "============================================================"

    if [[ -f "${ckpt_dir}/adapter_config.json" ]]; then
        echo "  SKIP TRAIN: adapter already exists at $ckpt_dir"
    else
        if [[ -d "$ckpt_dir" ]]; then
            echo "  WARN: stub checkpoint dir; removing"
            rm -rf "$ckpt_dir"
        fi
        echo "  --- TRAIN ---"
        local t0; t0=$(date +%s)
        if accelerate launch --num_processes="$NUM_PROCESSES" --mixed_precision=no \
            train.py --experiment "$yaml" --batch-size 2 --sft \
            2>&1 | tee "$log"; then
            local dt=$(( $(date +%s) - t0 ))
            echo "  TRAIN DONE: ${stem} in $((dt / 60))m $((dt % 60))s"
        else
            echo "  TRAIN FAILED for ${stem}; see $log"
            return 1
        fi
    fi

    echo "  --- EVAL ---"
    CUDA_VISIBLE_DEVICES=0 python3 eval.py --experiment "$yaml" --force \
        2>&1 | tee -a "$log"
}

start_time=$(date +%s)
# nanochat first — smaller, faster, fails fast on bugs
run_cell "nanochat_arc_r128" "nanochat" || true
run_cell "talkie_arc_r128"   "talkie"   || true
total=$(( $(date +%s) - start_time ))

echo
echo "============================================================="
echo "  ARC r=128 capacity experiment complete"
echo "============================================================="
echo "  Wall time:  $((total / 3600))h $((total % 3600 / 60))m $((total % 60))s"
echo

# --- Comparison table ---
echo "ARC accuracy comparison (lower is worse, 0.25 = chance):"
echo
printf "  %-35s %s\n" "Cell" "Accuracy"
printf "  %-35s %s\n" "-----------------------------------" "----------------------"
for name in talkie_arc_base talkie_arc talkie_arc_r128 \
            nanochat_arc_base nanochat_arc nanochat_arc_r128; do
    f="${POLICY_PRED_DATA_ROOT}/experiments/benchmark_${name}/eval.json"
    if [[ -f "$f" ]]; then
        line=$(python3 -c "
import json
with open('$f') as fp: d = json.load(fp)
r = d.get('results', {})
acc = r.get('pass_at_1', r.get('accuracy', None))
print(f'{acc:.4f} ({r.get(\"n_correct\", \"?\")}/{r.get(\"n_total\", \"?\")})' if acc is not None else 'PARSE_ERR')
" 2>/dev/null || echo "PARSE_ERR")
        printf "  %-35s %s\n" "$name" "$line"
    else
        printf "  %-35s %s\n" "$name" "MISSING"
    fi
done
