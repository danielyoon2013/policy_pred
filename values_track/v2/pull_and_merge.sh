#!/usr/bin/env bash
# Phase 2b — pull a pod's values eval.jsons + logs into the LOCAL archive on D:,
# then VERIFY completeness before that pod is torn down. Nothing is left only on
# an ephemeral pod.
#
# Run this LOCALLY (git-bash) once a pod's sweep prints VALUES_EVAL_DONE.
#
#   bash pull_and_merge.sh <arm> <ssh-host> [remote_out_root] [expected_years_csv]
#     <arm>              caselaw | swmgst
#     <ssh-host>         user@host (or a Host alias) with the pod
#     remote_out_root    default: \$HOME/pp_values_<arm>   (matches eval_values_battery.sh OUT_ROOT)
#     expected_years_csv default: 1943-2020  (e.g. "1972-2020" or "1972,1976,1980")
#
# Layout produced (canonical local archive):
#   D:/hist_LLM/policy_pred/values_v2/eval/<arm>/values_v2_<Y>_nanochat_<arm>_n10000/eval.json
#   D:/hist_LLM/policy_pred/values_v2/logs/<arm>/...
set -uo pipefail

ARM="${1:?usage: <arm> <ssh-host> [remote_out_root] [expected_years]}"
HOST="${2:?missing ssh host}"
REMOTE_ROOT="${3:-\$HOME/pp_values_${ARM}}"
YEARS_SPEC="${4:-1943-2020}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes}"

ARCHIVE="/d/hist_LLM/policy_pred/values_v2"
EVAL_DIR="$ARCHIVE/eval/$ARM"
LOG_DIR="$ARCHIVE/logs/$ARM"
mkdir -p "$EVAL_DIR" "$LOG_DIR"

echo "[pull] $ARM from $HOST:$REMOTE_ROOT"
# tar the eval.jsons remotely and stream into the local archive
ssh $SSH_OPTS "$HOST" "cd $REMOTE_ROOT/experiments 2>/dev/null && tar czf - values_v2_*_${ARM}_n10000/eval.json" \
  | tar xzf - -C "$EVAL_DIR" || { echo "[pull] eval pull FAILED"; exit 1; }
# logs (best-effort)
ssh $SSH_OPTS "$HOST" "cd /tmp/values_eval_logs 2>/dev/null && tar czf - values_v2_*_${ARM}_n10000.log 2>/dev/null" \
  | tar xzf - -C "$LOG_DIR" 2>/dev/null || echo "[pull] (no logs pulled)"

# expand the expected-years spec (ranges "a-b" and comma lists)
expand_years() {
  local spec="$1" out=()
  IFS=',' read -ra parts <<< "$spec"
  for p in "${parts[@]}"; do
    if [[ "$p" == *-* ]]; then out+=($(seq "${p%-*}" "${p#*-}")); else out+=("$p"); fi
  done
  echo "${out[@]}"
}

echo "[verify] checking every expected (arm,year) eval.json is present locally..."
missing=(); present=0
for y in $(expand_years "$YEARS_SPEC"); do
  f="$EVAL_DIR/values_v2_${y}_nanochat_${ARM}_n10000/eval.json"
  # a year with no adapter legitimately produces no eval — only flag if the adapter existed.
  if [[ -f "$f" ]]; then present=$((present+1)); else missing+=("$y"); fi
done
echo "[verify] present=$present  missing=${#missing[@]} ${missing[*]:0:20}"

# append to the manifest
MAN="$ARCHIVE/MANIFEST.md"
{ [[ -f "$MAN" ]] || echo "# values_v2 local archive — pull log";
  echo "- $ARM  from $HOST  years=$YEARS_SPEC  present=$present  missing=${#missing[@]}  ($(date))"; } >> "$MAN"

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "[verify] INCOMPLETE — do NOT tear down the pod; re-run the sweep for the missing years, then re-pull."
  exit 2
fi
echo "[verify] COMPLETE — all expected eval.jsons archived locally; pod is safe to tear down."
