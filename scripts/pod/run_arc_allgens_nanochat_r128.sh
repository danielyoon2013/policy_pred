#!/usr/bin/env bash
# Pod runbook: nanochat + ARC + all-gens, but with LoRA r=128.
# Variant of run_arc_allgens_nanochat.sh; reuses the same training data.
#
# Usage:
#   bash scripts/pod/run_arc_allgens_nanochat_r128.sh

set -uo pipefail

export NANOCHAT_WEIGHTS_DIR="${NANOCHAT_WEIGHTS_DIR:-${HOME}/nanochat_base}"
export POLICY_PRED_DATA_ROOT="${POLICY_PRED_DATA_ROOT:-${HOME}/policy_pred_data}"

NUM_PROCESSES="${NUM_PROCESSES:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [[ -z "$NUM_PROCESSES" || "$NUM_PROCESSES" -lt 1 ]]; then NUM_PROCESSES=4; fi

echo "============================================================="
echo "  Pod runbook: nanochat + ARC + all-gens + LoRA r=128"
echo "============================================================="
echo

if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

if [[ ! -f "data_links/benchmarks/arc_train_allgens.jsonl" ]]; then
    echo "ERROR: data_links/benchmarks/arc_train_allgens.jsonl missing"
    echo "  Should already exist from the previous run."
    exit 1
fi
echo "  arc_train_allgens.jsonl: $(wc -l < data_links/benchmarks/arc_train_allgens.jsonl) records"
echo

YAML="experiments/benchmark/nanochat_arc_allgens_r128.yaml"
EXP_NAME="benchmark_nanochat_arc_allgens_r128"
CKPT_DIR="${POLICY_PRED_DATA_ROOT}/experiments/${EXP_NAME}/checkpoint"
LOG="/tmp/${EXP_NAME}.log"

if [[ -f "${CKPT_DIR}/adapter_config.json" ]]; then
    echo "SKIP TRAIN: adapter already exists at $CKPT_DIR"
else
    if [[ -d "$CKPT_DIR" ]]; then
        rm -rf "$CKPT_DIR"
    fi
    echo "--- TRAIN ---"
    t0=$(date +%s)
    if accelerate launch --num_processes="$NUM_PROCESSES" --mixed_precision=no \
        train.py --experiment "$YAML" --batch-size 2 --sft \
        2>&1 | tee "$LOG"; then
        dt=$(( $(date +%s) - t0 ))
        echo "TRAIN DONE in $((dt / 60))m $((dt % 60))s"
    else
        echo "TRAIN FAILED; see $LOG"
        exit 1
    fi
fi

echo
echo "--- EVAL ---"
CUDA_VISIBLE_DEVICES=0 python3 eval.py --experiment "$YAML" --force \
    2>&1 | tee -a "$LOG"

echo
echo "============================================================="
echo "  Result comparison"
echo "============================================================="
printf "  %-40s %s\n" "Cell" "ARC Accuracy"
printf "  %-40s %s\n" "----------------------------------------" "----------------------"
for name in nanochat_arc_base nanochat_arc nanochat_arc_r128 nanochat_arc_allgens nanochat_arc_allgens_r128; do
    f="${POLICY_PRED_DATA_ROOT}/experiments/benchmark_${name}/eval.json"
    if [[ -f "$f" ]]; then
        line=$(python3 -c "
import json
with open('$f') as fp: d = json.load(fp)
r = d.get('results', {})
acc = r.get('pass_at_1', r.get('accuracy', None))
print(f'{acc:.4f} ({r.get(\"n_correct\", \"?\")}/{r.get(\"n_total\", \"?\")})' if acc is not None else 'PARSE_ERR')
" 2>/dev/null || echo "PARSE_ERR")
        printf "  %-40s %s\n" "$name" "$line"
    else
        printf "  %-40s %s\n" "$name" "MISSING"
    fi
done
