#!/usr/bin/env bash
# Pod runbook: 4 base-only benchmark evals — {Talkie, nanochat} × {ARC, GSM8K}.
#
# Pre-Phase-3 sanity check. NO training, NO LoRA — just probes each
# backend's base model on each benchmark to anchor the baseline numbers
# we'll compare LoRA-SFT scores against later.
#
# Prerequisites on the pod (set up by user before running this):
#   1. ~/policy_pred  — feat/matrix-experiments branch checked out + pulled
#   2. ~/talkie_base/  — Talkie weights (from prior session) OR override
#      TALKIE_WEIGHTS_DIR
#   3. ~/nanochat_base/ — nanochat weights (extract nanochat_pod_bundle.zip
#      here) OR override NANOCHAT_WEIGHTS_DIR
#   4. ~/policy_pred/data_links/eval/  — arc_challenge_test.jsonl +
#      gsm_mc.jsonl (extract from nanochat_pod_bundle.zip into here)
#   5. transformers==4.46.3, peft==0.13.2 pinned in venv
#   6. .venv activated
#
# Usage:
#   bash scripts/pod/run_benchmark_base_evals.sh
#
# Output: 4 eval.json files under
#   $POLICY_PRED_DATA_ROOT/experiments/benchmark_<name>_base/eval.json
#
# Wall time: ~10-15 min on 4× H100 (single GPU per eval, run serial).
#   Talkie 13B: ~5 min/eval, nanochat 1.36B: ~30 sec/eval.

set -uo pipefail

export TALKIE_WEIGHTS_DIR="${TALKIE_WEIGHTS_DIR:-${HOME}/talkie_base}"
export NANOCHAT_WEIGHTS_DIR="${NANOCHAT_WEIGHTS_DIR:-${HOME}/nanochat_base}"
export POLICY_PRED_DATA_ROOT="${POLICY_PRED_DATA_ROOT:-${HOME}/policy_pred_data}"

echo "============================================================="
echo "  Pod runbook: benchmark base-only evals (4 cells)"
echo "============================================================="
echo "  Talkie weights:    $TALKIE_WEIGHTS_DIR"
echo "  Nanochat weights:  $NANOCHAT_WEIGHTS_DIR"
echo "  Data root:         $POLICY_PRED_DATA_ROOT"
echo

# --- venv ---
if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# --- Sanity checks ---
miss=0
if [[ ! -d "$TALKIE_WEIGHTS_DIR" ]] || [[ ! -f "$TALKIE_WEIGHTS_DIR/final.ckpt" ]]; then
    echo "WARN: Talkie weights not found at $TALKIE_WEIGHTS_DIR"
    echo "      Talkie evals will be skipped."
    HAS_TALKIE=0
else
    HAS_TALKIE=1
fi
if [[ ! -d "$NANOCHAT_WEIGHTS_DIR" ]] || \
   ! ls "$NANOCHAT_WEIGHTS_DIR/base_checkpoints"/d* >/dev/null 2>&1; then
    echo "WARN: Nanochat weights not found at $NANOCHAT_WEIGHTS_DIR/base_checkpoints/d*"
    echo "      Nanochat evals will be skipped."
    HAS_NANOCHAT=0
else
    HAS_NANOCHAT=1
fi
if [[ ! -f "data_links/eval/arc_challenge_test.jsonl" ]] || \
   [[ ! -f "data_links/eval/gsm_mc.jsonl" ]]; then
    echo "ERROR: missing data_links/eval/{arc_challenge_test,gsm_mc}.jsonl"
    echo "       Extract them from nanochat_pod_bundle.zip into data_links/eval/."
    exit 1
fi

if [[ "$HAS_TALKIE" -eq 0 ]] && [[ "$HAS_NANOCHAT" -eq 0 ]]; then
    echo "ERROR: no backend weights available. Stopping."
    exit 1
fi

# --- Run evals (pin to GPU 0 — each eval is single-GPU) ---
export CUDA_VISIBLE_DEVICES=0

start_time=$(date +%s)
n_done=0
n_failed=0

run_eval() {
    local yaml="$1"
    local label="$2"
    echo
    echo "============================================================"
    echo "  EVAL: $label"
    echo "============================================================"
    yr_start=$(date +%s)
    if python3 eval.py \
        --experiment "$yaml" \
        --no-adapter \
        --force; then
        elapsed=$(( $(date +%s) - yr_start ))
        echo "  $label: OK (${elapsed}s)"
        n_done=$((n_done + 1))
    else
        echo "  $label: FAILED"
        n_failed=$((n_failed + 1))
    fi
}

if [[ "$HAS_TALKIE" -eq 1 ]]; then
    run_eval "experiments/benchmark/talkie_arc_base.yaml"   "talkie + ARC (base)"
    run_eval "experiments/benchmark/talkie_gsm8k_base.yaml" "talkie + GSM8K (base)"
fi

if [[ "$HAS_NANOCHAT" -eq 1 ]]; then
    run_eval "experiments/benchmark/nanochat_arc_base.yaml"   "nanochat + ARC (base)"
    run_eval "experiments/benchmark/nanochat_gsm8k_base.yaml" "nanochat + GSM8K (base)"
fi

total_elapsed=$(( $(date +%s) - start_time ))

echo
echo "============================================================="
echo "  Base-only eval matrix complete"
echo "============================================================="
echo "  Wall time:     $((total_elapsed / 60))m $((total_elapsed % 60))s"
echo "  Cells done:    $n_done / 4"
echo "  Cells failed:  $n_failed"
echo
echo "Results:"
for name in talkie_arc_base talkie_gsm8k_base nanochat_arc_base nanochat_gsm8k_base; do
    f="${POLICY_PRED_DATA_ROOT}/experiments/benchmark_${name}/eval.json"
    if [[ -f "$f" ]]; then
        acc=$(python3 -c "
import json
with open('$f') as fp:
    d = json.load(fp)
r = d.get('results', {})
print(f\"{r.get('pass_at_1', r.get('accuracy', '?'))} ({r.get('n_correct', '?')}/{r.get('n_total', '?')})\")
" 2>/dev/null || echo "PARSE_ERROR")
        printf "  %-30s  %s\n" "$name" "$acc"
    else
        printf "  %-30s  MISSING\n" "$name"
    fi
done
echo
echo "Pull results back to local:"
echo "  tar czf /tmp/benchmark_base_evals.tgz \\"
echo "      $POLICY_PRED_DATA_ROOT/experiments/benchmark_*_base/eval.json"
