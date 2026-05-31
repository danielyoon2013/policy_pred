#!/usr/bin/env bash
# Pod runbook: variant-aware eval of the rolling-window matrix, GPU-pinned.
#
# Twin of train_window_matrix.sh: one eval.py per (year, X[, _sft]) cell, sharded
# across GPUs via a sliding job pool. Each cell's YAML carries window_years:10, so
# every policy is scored by year-models E-10..E+10 — the n-shape. eval.json lands
# at $POLICY_PRED_DATA_ROOT/experiments/<name>/eval.json (name-routed).
#
# Usage:
#   bash scripts/pod/eval_window_matrix.sh <backend> <spec> <stage> [start_year] [end_year]
#   JOBS_PER_GPU=2 bash scripts/pod/eval_window_matrix.sh nanochat roll10 lora
#   JOBS_PER_GPU=2 bash scripts/pod/eval_window_matrix.sh nanochat roll10 sft
#
# Env: X_LEVELS, JOBS_PER_GPU (1 talkie / 2 nanochat). Idempotent (skips a cell
# whose eval.json already exists — checked in bash so we never load the base to
# no-op).
set -uo pipefail

BACKEND="${1:?usage: <backend> <spec> <stage> [start_year] [end_year]}"
SPEC="${2:?missing spec}"
STAGE="${3:?missing stage (lora|sft)}"
START_YEAR="${4:-1931}"
END_YEAR="${5:-2019}"

X_LEVELS="${X_LEVELS:-5000 10000 20000 40000}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

export TALKIE_WEIGHTS_DIR="${TALKIE_WEIGHTS_DIR:-${HOME}/talkie_base}"
export NANOCHAT_WEIGHTS_DIR="${NANOCHAT_WEIGHTS_DIR:-${HOME}/nanochat_base}"
export POLICY_PRED_DATA_ROOT="${POLICY_PRED_DATA_ROOT:-${HOME}/policy_pred_data}"

EXPS_DIR="${POLICY_PRED_DATA_ROOT}/experiments"
YAML_DIR="experiments/windows"
LOG_DIR="${LOG_DIR:-/tmp/window_eval_logs}"
mkdir -p "$LOG_DIR"

if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

mapfile -t GPUS < <(nvidia-smi -L 2>/dev/null | sed -E 's/GPU ([0-9]+):.*/\1/')
if [[ ${#GPUS[@]} -eq 0 ]]; then GPUS=(0); fi

echo "============================================================="
echo "  Window eval: backend=$BACKEND spec=$SPEC stage=$STAGE"
echo "  Years ${START_YEAR}..${END_YEAR}  X=[${X_LEVELS}]  GPUs=${GPUS[*]} x${JOBS_PER_GPU}"
echo "============================================================="

free=()
for g in "${GPUS[@]}"; do for ((j=0; j<JOBS_PER_GPU; j++)); do free+=("$g"); done; done

queue=()
for y in $(seq "$START_YEAR" "$END_YEAR"); do
    for x in $X_LEVELS; do queue+=("$y $x"); done
done
n_queue=${#queue[@]}

declare -A pid_gpu=() pid_name=()
qi=0; n_done=0; n_fail=0; n_skip=0
start_time=$(date +%s)
sftflag=$([[ "$STAGE" == "sft" ]] && echo --sft || echo "")

run_cell() {
    local gpu="$1" y="$2" x="$3" name log
    name=$(python -m windowing name --year "$y" --backend "$BACKEND" --spec "$SPEC" \
            --per-year "$x" $sftflag)
    log="${LOG_DIR}/${name}.log"
    if [[ -f "${EXPS_DIR}/${name}/eval.json" ]]; then return 2; fi          # done
    if [[ ! -f "${EXPS_DIR}/${name}/checkpoint/adapter_config.json" ]]; then
        echo "no adapter for $name" > "$log"; return 3                       # not trained
    fi
    CUDA_VISIBLE_DEVICES="$gpu" python eval.py \
        --experiment "${YAML_DIR}/${name}.yaml" \
        --evaluator policy_battery_variants --force > "$log" 2>&1
    [[ -f "${EXPS_DIR}/${name}/eval.json" ]] && return 0 || return 1
}

reap() {
    wait -n 2>/dev/null
    for pid in "${!pid_gpu[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"; local rc=$?
            free+=("${pid_gpu[$pid]}"); local nm="${pid_name[$pid]}"
            unset 'pid_gpu[$pid]' 'pid_name[$pid]'
            case $rc in
                0) n_done=$((n_done+1)); st="OK";;
                2) n_skip=$((n_skip+1)); st="SKIP(done)";;
                3) n_skip=$((n_skip+1)); st="SKIP(untrained)";;
                *) n_fail=$((n_fail+1)); st="FAIL";;
            esac
            echo "  [$((n_done+n_fail+n_skip))/$n_queue] $nm $st ($(( $(date +%s) - start_time ))s)"
        fi
    done
}

while [[ $qi -lt $n_queue || ${#pid_gpu[@]} -gt 0 ]]; do
    while [[ ${#free[@]} -gt 0 && $qi -lt $n_queue ]]; do
        gpu="${free[0]}"; free=("${free[@]:1}")
        read -r y x <<< "${queue[$qi]}"; qi=$((qi+1))
        name=$(python -m windowing name --year "$y" --backend "$BACKEND" --spec "$SPEC" \
                --per-year "$x" $sftflag)
        run_cell "$gpu" "$y" "$x" &
        pid=$!; pid_gpu[$pid]="$gpu"; pid_name[$pid]="$name"
    done
    [[ ${#pid_gpu[@]} -gt 0 ]] && reap
done

total=$(( $(date +%s) - start_time ))
echo
echo "  eval '$STAGE' done: $n_done ok, $n_skip skipped, $n_fail failed, ${total}s"
echo "Pull results:  cd $EXPS_DIR && tar czf /tmp/window_${BACKEND}_${SPEC}_${STAGE}_evals.tgz \\"
echo "    policy_*_${BACKEND}_${SPEC}_n*$([[ "$STAGE" == "sft" ]] && echo _sft)/eval.json"
