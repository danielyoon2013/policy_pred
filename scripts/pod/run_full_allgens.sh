#!/usr/bin/env bash
# Pod runbook: full all-generators LoRA-SFT experiment on ARC + RACE.
#
# Per backend (nanochat first, then Talkie):
#   1. Train ONE LoRA-SFT adapter on the full A-G mix (1.27M records)
#   2. Eval that adapter on ARC-Challenge (1144 items)
#   3. Eval that same adapter on RACE (1240 items: 360 middle + 880 high)
#
# Reuses one trained adapter per backend across both benchmarks via
# eval.py's --adapter override.
#
# Prereqs on pod:
#   - data_links/benchmarks/train_full_allgens.jsonl  (1.27M records)
#   - data_links/eval/race_test.jsonl                 (1240 RACE items)
#   - data_links/eval/arc_challenge_test.jsonl        (already there)
#   - Talkie + nanochat weights (already there)
#
# Wall time estimate:
#   nanochat (1.36B) train: ~3-4 hr  + 2 evals: ~5 min  -> ~3-4 hr
#   talkie (13B)    train: ~15-18 hr + 2 evals: ~10 min -> ~15-18 hr
#   Total:                                              ~18-22 hr
#
# Usage:
#   bash scripts/pod/run_full_allgens.sh
#
# Tip: tmux/nohup this. Logs land in /tmp/benchmark_full_allgens_*.log

set -uo pipefail

export TALKIE_WEIGHTS_DIR="${TALKIE_WEIGHTS_DIR:-${HOME}/talkie_base}"
export NANOCHAT_WEIGHTS_DIR="${NANOCHAT_WEIGHTS_DIR:-${HOME}/nanochat_base}"
export POLICY_PRED_DATA_ROOT="${POLICY_PRED_DATA_ROOT:-${HOME}/policy_pred_data}"

NUM_PROCESSES="${NUM_PROCESSES:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [[ -z "$NUM_PROCESSES" || "$NUM_PROCESSES" -lt 1 ]]; then NUM_PROCESSES=4; fi

echo "============================================================="
echo "  Pod runbook: full A-G LoRA-SFT on ARC + RACE"
echo "============================================================="
echo "  GPUs:              $NUM_PROCESSES"
echo "  Data root:         $POLICY_PRED_DATA_ROOT"
echo

if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# --- Data sanity ---
TRAIN="data_links/benchmarks/train_full_allgens.jsonl"
RACE="data_links/eval/race_test.jsonl"
ARC="data_links/eval/arc_challenge_test.jsonl"
for f in "$TRAIN" "$RACE" "$ARC"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing $f"
        echo "  Upload + extract benchmark_full_allgens.zip first."
        exit 1
    fi
done
echo "  $TRAIN: $(wc -l < "$TRAIN") records (expect ~1275260)"
echo "  $RACE:  $(wc -l < "$RACE") records (expect 1240)"
echo "  $ARC:   $(wc -l < "$ARC") records (expect 1144)"
echo

run_backend() {
    local backend="$1"   # "nanochat" | "talkie"
    local arc_yaml="experiments/benchmark/${backend}_full_allgens.yaml"
    local race_yaml="experiments/benchmark/${backend}_full_allgens_race.yaml"
    local arc_exp="benchmark_${backend}_full_allgens"
    local race_exp="benchmark_${backend}_full_allgens_race"
    local ckpt_dir="${POLICY_PRED_DATA_ROOT}/experiments/${arc_exp}/checkpoint"
    local log="/tmp/${arc_exp}.log"

    echo
    echo "==============================================================="
    echo "  BACKEND: ${backend}"
    echo "==============================================================="

    # --- Train (idempotent) ---
    if [[ -f "${ckpt_dir}/adapter_config.json" ]]; then
        echo "  SKIP TRAIN: adapter exists at $ckpt_dir"
    else
        if [[ -d "$ckpt_dir" ]]; then
            echo "  WARN: stub checkpoint dir; removing"
            rm -rf "$ckpt_dir"
        fi
        echo "  --- TRAIN ${backend} ---"
        t0=$(date +%s)
        if accelerate launch --num_processes="$NUM_PROCESSES" --mixed_precision=no \
            train.py --experiment "$arc_yaml" --batch-size 2 --sft \
            2>&1 | tee "$log"; then
            dt=$(( $(date +%s) - t0 ))
            echo "  TRAIN DONE: ${backend} in $((dt / 3600))h $((dt % 3600 / 60))m"
        else
            echo "  TRAIN FAILED for ${backend}; see $log"
            return 1
        fi
    fi

    # --- Eval ARC ---
    echo "  --- EVAL ${backend} on ARC ---"
    CUDA_VISIBLE_DEVICES=0 python3 eval.py --experiment "$arc_yaml" --force \
        2>&1 | tee -a "$log"

    # --- Eval RACE (same adapter, different test set) ---
    echo "  --- EVAL ${backend} on RACE ---"
    CUDA_VISIBLE_DEVICES=0 python3 eval.py \
        --experiment "$race_yaml" \
        --adapter "$ckpt_dir" \
        --force \
        2>&1 | tee -a "$log"
}

start_time=$(date +%s)
run_backend "nanochat" || true
run_backend "talkie"   || true
total=$(( $(date +%s) - start_time ))

echo
echo "==============================================================="
echo "  Run complete"
echo "==============================================================="
echo "  Wall time:  $((total / 3600))h $((total % 3600 / 60))m $((total % 60))s"
echo

# --- Final comparison ---
echo "Comparison table — accuracy by (backend, benchmark, recipe):"
echo
printf "  %-45s %s\n" "Cell" "Accuracy"
printf "  %-45s %s\n" "---------------------------------------------" "----------------------"
for name in \
    nanochat_arc_base nanochat_arc nanochat_arc_r128 nanochat_arc_allgens \
    nanochat_full_allgens nanochat_full_allgens_race \
    talkie_arc_base talkie_arc \
    talkie_full_allgens talkie_full_allgens_race ; do
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
