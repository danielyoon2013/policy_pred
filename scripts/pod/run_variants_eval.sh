#!/usr/bin/env bash
# Pod runbook: re-eval the existing 1931-1935 LoRA adapters using the
# variant-aware policy_battery_variants evaluator. Produces an SS-1935
# trajectory with mean±std error bars on the centered-likert scale.
#
# Prerequisites (must be done BEFORE running this script):
#   1. ~/policy_question_variants.zip uploaded via WinSCP/scp (0.48 MB)
#   2. ~/policy_pred git checked out with phase-1 + phase-2-prep commits
#      (commits a321dc6 and 770be45 or later)
#   3. ~/talkie_base/ + ~/policy_pred_data/experiments/policy_193{1..5}_*/
#      checkpoints already exist (from the original 1k-seed training run)
#
# Usage:
#   bash scripts/pod/run_variants_eval.sh
#
# After it finishes, scp the eval.json files back to local for plotting:
#   tar czf /tmp/variants_evals.tgz \
#       policy_pred_data/experiments/policy_193{1,2,3,4,5}_*/eval.json \
#       policy_pred_data/experiments/policy_1931_base/eval.json
#   # then WinSCP /tmp/variants_evals.tgz to local C:/tmp/
#
# Wall time: ~30-60 min total (6 evals × ~5-10 min each on a single A100).
#
set -euo pipefail

# --- Settings ---
EXPERIMENTS=(
    "policy_1931_base"
    "policy_1931_naive"
    "policy_1932_naive"
    "policy_1933_naive"
    "policy_1934_naive"
    "policy_1935_naive"
)
POLICY_FOR_SUMMARY="social_security_1935"
VARIANTS_ZIP="${HOME}/policy_question_variants.zip"
VARIANTS_DIR="data_artifacts/question_variants"
EXPECTED_VARIANT_FILES=211

# --- Env vars ---
export TALKIE_WEIGHTS_DIR="${TALKIE_WEIGHTS_DIR:-${HOME}/talkie_base}"
export POLICY_PRED_DATA_ROOT="${POLICY_PRED_DATA_ROOT:-${HOME}/policy_pred_data}"

echo "============================================================="
echo "  Pod runbook: variant-aware re-eval of 1931-1935 chain"
echo "============================================================="
echo
echo "  Talkie weights:   $TALKIE_WEIGHTS_DIR"
echo "  Data root:        $POLICY_PRED_DATA_ROOT"
echo "  Variants zip:     $VARIANTS_ZIP"
echo "  Variants dir:     $VARIANTS_DIR (under repo)"
echo "  Experiments:      ${#EXPERIMENTS[@]}"
echo

# --- Sanity checks ---
echo "--- Sanity checks ---"

if [[ ! -d "$TALKIE_WEIGHTS_DIR" ]]; then
    echo "ERROR: TALKIE_WEIGHTS_DIR ($TALKIE_WEIGHTS_DIR) does not exist."
    echo "       Make sure Talkie weights are downloaded."
    exit 1
fi
if [[ ! -f "$TALKIE_WEIGHTS_DIR/final.ckpt" ]] || [[ ! -f "$TALKIE_WEIGHTS_DIR/vocab.txt" ]]; then
    echo "ERROR: Talkie weights incomplete (need final.ckpt + vocab.txt)."
    exit 1
fi
echo "  Talkie weights: OK"

if [[ ! -d "$POLICY_PRED_DATA_ROOT" ]]; then
    echo "ERROR: POLICY_PRED_DATA_ROOT ($POLICY_PRED_DATA_ROOT) does not exist."
    exit 1
fi
echo "  Data root:      OK"

missing_checkpoint=0
for exp in "${EXPERIMENTS[@]}"; do
    [[ "$exp" == "policy_1931_base" ]] && continue   # base eval doesn't need a checkpoint
    if [[ ! -d "$POLICY_PRED_DATA_ROOT/experiments/$exp/checkpoint" ]]; then
        echo "WARN: missing adapter checkpoint for $exp"
        missing_checkpoint=$((missing_checkpoint + 1))
    fi
done
if [[ "$missing_checkpoint" -gt 0 ]]; then
    echo "ERROR: $missing_checkpoint adapter checkpoint(s) missing. Train them first or"
    echo "       remove the corresponding entries from EXPERIMENTS above."
    exit 1
fi
echo "  Adapter checkpoints: OK (5/5 trained + base)"

# --- Activate venv ---
if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "  venv:            activated"
fi

# --- Pull latest code ---
echo
echo "--- git pull (ensures Phase 1 code is present) ---"
git pull --ff-only

# --- Extract variants ---
echo
echo "--- Extracting question variants ---"
if [[ ! -f "$VARIANTS_ZIP" ]]; then
    echo "ERROR: $VARIANTS_ZIP not found. Upload it via WinSCP/scp first."
    exit 1
fi

mkdir -p "$VARIANTS_DIR"
n_before=$(find "$VARIANTS_DIR" -maxdepth 1 -name '*.jsonl' | wc -l)
if [[ "$n_before" -ge "$EXPECTED_VARIANT_FILES" ]]; then
    echo "  Variants already extracted ($n_before files); skipping unzip."
else
    python -m zipfile -e "$VARIANTS_ZIP" "$VARIANTS_DIR"
fi
n_files=$(find "$VARIANTS_DIR" -maxdepth 1 -name '*.jsonl' | wc -l)
echo "  Found $n_files variant files (expected $EXPECTED_VARIANT_FILES)"
if [[ "$n_files" -lt "$EXPECTED_VARIANT_FILES" ]]; then
    echo "WARN: fewer variant files than expected. The eval will only score policies"
    echo "      with variant files present; others will be skipped."
fi

# --- Re-eval each experiment ---
echo
echo "--- Running policy_battery_variants on ${#EXPERIMENTS[@]} adapters ---"
echo "    (uses --force to overwrite the old single-question eval.json)"
echo

# For base eval, also need --no-adapter so eval.py doesn't try to load one.
for exp in "${EXPERIMENTS[@]}"; do
    echo "================================================"
    echo "  Eval: $exp"
    echo "================================================"
    extra_args=""
    if [[ "$exp" == *"_base" ]]; then
        extra_args="--no-adapter"
    fi
    python3 eval.py \
        --experiment "experiments/${exp}.yaml" \
        --evaluator policy_battery_variants \
        --force \
        $extra_args
    echo
done

# --- Print trajectory ---
echo
echo "============================================================="
echo "  Final SS-1935 trajectory (mean ± std across 100 variants)"
echo "============================================================="
python3 scripts/summarize_policy_trajectory.py \
    --policy "$POLICY_FOR_SUMMARY" \
    --glob "$POLICY_PRED_DATA_ROOT/experiments/policy_*/eval.json"

# --- Final hint ---
echo
echo "------------------------------------------------------------"
echo "  Done. To pull results back to local for plotting:"
echo "    cd $POLICY_PRED_DATA_ROOT/experiments"
echo "    tar czf /tmp/variants_evals.tgz \\"
echo "        policy_1931_base/eval.json \\"
echo "        policy_193{1,2,3,4,5}_naive/eval.json"
echo "    # then WinSCP /tmp/variants_evals.tgz to local C:/tmp/"
echo "------------------------------------------------------------"
