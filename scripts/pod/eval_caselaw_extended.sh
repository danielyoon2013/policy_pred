#!/usr/bin/env bash
# Phase 0: re-evaluate the EXISTING caselaw nanochat adapters with the extended
# rel_min=-10 / rel_max=+20 window AND both yes_no + yes_no_cot, on a pod.
#
# No retraining — this only re-runs eval against adapters already uploaded to
# $POLICY_PRED_DATA_ROOT/experiments/policy_<Y>_nanochat_roll10_n<PER_YEAR>/checkpoint
# (the round2 caselaw sweep). It (1) regenerates each year's eval YAML with the
# asymmetric rel window + dual modes, then (2) runs eval_window_matrix.sh at that X.
#
# The eval.json it writes is the EXTENDED-window + reasoning eval; pull it into a
# fresh local dir (docs/findings/2026-06/caselaw_extended_eval/) — it does not
# touch the original +/-10 eval.json archived on D:.
#
# Usage:
#   bash scripts/pod/eval_caselaw_extended.sh [start_year] [end_year]
#   VARIANTS_CAP=12 bash scripts/pod/eval_caselaw_extended.sh 1931 2020
set -uo pipefail

START_YEAR="${1:-1931}"
END_YEAR="${2:-2020}"
PER_YEAR="${PER_YEAR:-10000}"
REL_MIN="${REL_MIN:--10}"
REL_MAX="${REL_MAX:-20}"
MODES="${MODES:-yes_no,yes_no_cot}"
VARIANTS_CAP="${VARIANTS_CAP:-12}"        # cap paraphrases/policy (CoT gen is slow)
COT_TOKENS="${COT_TOKENS:-256}"

export POLICY_PRED_DATA_ROOT="${POLICY_PRED_DATA_ROOT:-${HOME}/policy_pred_data}"
export NANOCHAT_WEIGHTS_DIR="${NANOCHAT_WEIGHTS_DIR:-${HOME}/nanochat_base}"
YAML_DIR="experiments/windows"
mkdir -p "$YAML_DIR"
[[ -d .venv ]] && source .venv/bin/activate

echo "============================================================="
echo "  Phase 0: caselaw extended-window + reasoning re-eval"
echo "  years ${START_YEAR}..${END_YEAR}  X=${PER_YEAR}  rel=[${REL_MIN},${REL_MAX}]"
echo "  modes=${MODES}  variants_cap=${VARIANTS_CAP}"
echo "============================================================="

# 1) Regenerate each year's eval YAML with the extended window + dual modes.
for y in $(seq "$START_YEAR" "$END_YEAR"); do
    python -m windowing yaml --year "$y" --backend nanochat --spec roll10 \
        --per-year "$PER_YEAR" --modes "$MODES" \
        --rel-min="$REL_MIN" --rel-max="$REL_MAX" \
        --variants-per-policy "$VARIANTS_CAP" --cot-max-new-tokens "$COT_TOKENS" \
        --out-dir "$YAML_DIR" >/dev/null || echo "  (yaml gen failed for $y)"
done
echo "Regenerated eval YAMLs."

# 2) Run the sharded window-matrix eval at this single X level.
X_LEVELS="$PER_YEAR" bash scripts/pod/eval_window_matrix.sh nanochat roll10 lora \
    "$START_YEAR" "$END_YEAR"

echo
echo "Pull results:"
echo "  cd \$POLICY_PRED_DATA_ROOT/experiments"
echo "  tar czf /tmp/caselaw_extended_evals.tgz policy_*_nanochat_roll10_n${PER_YEAR}/eval.json"
