#!/usr/bin/env bash
# Pod runbook: GSM8K-MC eval on the two already-trained full_allgens adapters.
# No training — just two eval runs reusing existing checkpoints via --adapter.
#
# Usage:
#   bash scripts/pod/run_full_allgens_gsm.sh
#
# Wall time: ~3-5 min total (nanochat ~30 sec, Talkie ~2-3 min).

set -uo pipefail

export TALKIE_WEIGHTS_DIR="${TALKIE_WEIGHTS_DIR:-${HOME}/talkie_base}"
export NANOCHAT_WEIGHTS_DIR="${NANOCHAT_WEIGHTS_DIR:-${HOME}/nanochat_base}"
export POLICY_PRED_DATA_ROOT="${POLICY_PRED_DATA_ROOT:-${HOME}/policy_pred_data}"

echo "============================================================="
echo "  Pod runbook: GSM8K-MC eval on full_allgens adapters"
echo "============================================================="
echo

if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# --- Verify GSM8K test set + adapter checkpoints present ---
GSM="data_links/eval/gsm_mc.jsonl"
NANOCHAT_CKPT="${POLICY_PRED_DATA_ROOT}/experiments/benchmark_nanochat_full_allgens/checkpoint"
TALKIE_CKPT="${POLICY_PRED_DATA_ROOT}/experiments/benchmark_talkie_full_allgens/checkpoint"

for f in "$GSM"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing $f"; exit 1
    fi
done
echo "  $GSM: $(wc -l < "$GSM") records (expect 1319)"

for d in "$NANOCHAT_CKPT" "$TALKIE_CKPT"; do
    if [[ ! -f "$d/adapter_config.json" ]]; then
        echo "ERROR: missing adapter at $d/adapter_config.json"
        echo "  Did the full_allgens training finish for that backend?"
        exit 1
    fi
done
echo "  nanochat adapter: $NANOCHAT_CKPT  ✓"
echo "  talkie   adapter: $TALKIE_CKPT     ✓"
echo

run_eval() {
    local backend="$1"
    local yaml="experiments/benchmark/${backend}_full_allgens_gsm.yaml"
    local ckpt="${POLICY_PRED_DATA_ROOT}/experiments/benchmark_${backend}_full_allgens/checkpoint"
    local log="/tmp/benchmark_${backend}_full_allgens_gsm.log"
    echo "----------------------------------------------------------"
    echo "  EVAL: ${backend} + full_allgens adapter -> GSM8K-MC"
    echo "----------------------------------------------------------"
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=0 python3 eval.py \
        --experiment "$yaml" \
        --adapter "$ckpt" \
        --force \
        2>&1 | tee "$log"
    dt=$(( $(date +%s) - t0 ))
    echo "  done in $((dt / 60))m $((dt % 60))s"
    echo
}

run_eval nanochat
run_eval talkie

# --- Final comparison table ---
echo
echo "==============================================================="
echo "  Updated comparison table (now including GSM8K-MC)"
echo "==============================================================="
printf "  %-45s %s\n" "Cell" "Accuracy"
printf "  %-45s %s\n" "---------------------------------------------" "----------------------"
for name in \
    nanochat_arc_base nanochat_arc nanochat_arc_allgens \
    nanochat_full_allgens nanochat_full_allgens_race nanochat_full_allgens_gsm \
    talkie_arc_base talkie_arc \
    talkie_full_allgens talkie_full_allgens_race talkie_full_allgens_gsm ; do
    f="${POLICY_PRED_DATA_ROOT}/experiments/benchmark_${name}/eval.json"
    if [[ -f "$f" ]]; then
        line=$(python3 -c "
import json
with open('$f') as fp: d = json.load(fp)
r = d.get('results', {})
acc = r.get('pass_at_1', r.get('accuracy', None))
print(f'{acc:.4f} ({r.get(\"n_correct\", \"?\")}/{r.get(\"n_total\", \"?\")})' if acc is not None else 'PARSE_ERR')
" 2>/dev/null || echo "PARSE_ERR")
        printf "  %-45s %s\n" "$name" "$line"
    else
        printf "  %-45s %s\n" "$name" "MISSING"
    fi
done
