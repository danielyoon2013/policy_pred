#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/workspace/continuous_d23_rel20}"
CODE="${CODE:-/workspace/policy_pred}"
POOL_DIR="${POOL_DIR:-${ROOT}/pools}"
BASE_ROOT="${BASE_ROOT:-${ROOT}/bases}"
CONTROL_DATA_ROOT="${CONTROL_DATA_ROOT:-${ROOT}/control_data}"
CONT_DATA_ROOT="${CONT_DATA_ROOT:-${ROOT}/continuous_data}"
SCRATCH="${SCRATCH:-${ROOT}/scratch}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
YAML_ROOT="${YAML_ROOT:-${ROOT}/yamls}"
PER_YEAR="${PER_YEAR:-10000}"
SPEC="${SPEC:-roll10}"
REL_MIN="${REL_MIN:--20}"
REL_MAX="${REL_MAX:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

mkdir -p "$SCRATCH" "$LOG_DIR" "$YAML_ROOT" "$CONT_DATA_ROOT" "$CONTROL_DATA_ROOT"

if [[ ! -d "$CODE" ]]; then
  echo "missing CODE dir: $CODE" >&2
  exit 1
fi
cd "$CODE" || exit 1

mapfile -t GPUS < <(nvidia-smi -L 2>/dev/null | sed -E 's/GPU ([0-9]+):.*/\1/')
if [[ ${#GPUS[@]} -eq 0 ]]; then GPUS=(0); fi

name_for() {
  local year="$1"
  python -m windowing name --year "$year" --backend nanochat --spec "$SPEC" --per-year "$PER_YEAR"
}

write_yaml() {
  local year="$1" yaml_dir="$2"
  python -m windowing yaml --year "$year" --backend nanochat --spec "$SPEC" \
    --per-year "$PER_YEAR" --modes yes_no,likert5 \
    --rel-min "$REL_MIN" --rel-max "$REL_MAX" \
    --no-chat-format --out-dir "$yaml_dir"
}

run_continuous_year() {
  local gpu="$1" year="$2"
  local name yaml_dir yaml data log base
  name="$(name_for "$year")"
  yaml_dir="${YAML_ROOT}/continuous"
  yaml="$(write_yaml "$year" "$yaml_dir" | tail -n 1)"
  data="${SCRATCH}/${name}.jsonl"
  log="${LOG_DIR}/continuous_${year}.log"
  base="${BASE_ROOT}/${year}"

  if [[ ! -d "$base/base_checkpoints/d23" ]]; then
    echo "missing staged base for ${year}: $base" | tee "$log"
    return 1
  fi

  export POLICY_PRED_DATA_ROOT="$CONT_DATA_ROOT"
  export NANOCHAT_WEIGHTS_DIR="$base"

  if [[ ! -f "${CONT_DATA_ROOT}/experiments/${name}/checkpoint/adapter_config.json" ]]; then
    echo "[continuous $year] assemble" > "$log"
    python -m windowing assemble --year "$year" --spec "$SPEC" --per-year "$PER_YEAR" \
      --pool-dir "$POOL_DIR" --out "$data" >> "$log" 2>&1 || return 1
    echo "[continuous $year] train on gpu $gpu" >> "$log"
    CUDA_VISIBLE_DEVICES="$gpu" python train.py --experiment "$yaml" \
      --data "$data" --batch-size "$BATCH_SIZE" >> "$log" 2>&1 || return 1
    rm -f "$data"
  else
    echo "[continuous $year] train skip, adapter exists" > "$log"
  fi

  if [[ ! -f "${CONT_DATA_ROOT}/experiments/${name}/eval.json" ]]; then
    echo "[continuous $year] eval on gpu $gpu" >> "$log"
    CUDA_VISIBLE_DEVICES="$gpu" python eval.py --experiment "$yaml" \
      --evaluator policy_battery_variants --force >> "$log" 2>&1 || return 1
  else
    echo "[continuous $year] eval skip, eval exists" >> "$log"
  fi
}

run_control_year() {
  local gpu="$1" year="$2"
  local name yaml_dir yaml log
  name="$(name_for "$year")"
  yaml_dir="${YAML_ROOT}/control"
  yaml="$(write_yaml "$year" "$yaml_dir" | tail -n 1)"
  log="${LOG_DIR}/control_${year}.log"

  export POLICY_PRED_DATA_ROOT="$CONTROL_DATA_ROOT"
  export NANOCHAT_WEIGHTS_DIR="${ROOT}/control_base"

  if [[ ! -f "${CONTROL_DATA_ROOT}/experiments/${name}/checkpoint/adapter_config.json" ]]; then
    echo "missing control adapter for ${year}: ${CONTROL_DATA_ROOT}/experiments/${name}/checkpoint" | tee "$log"
    return 1
  fi

  if [[ ! -f "${CONTROL_DATA_ROOT}/experiments/${name}/eval.json" ]]; then
    echo "[control $year] eval on gpu $gpu" > "$log"
    CUDA_VISIBLE_DEVICES="$gpu" python eval.py --experiment "$yaml" \
      --evaluator policy_battery_variants --force >> "$log" 2>&1 || return 1
  else
    echo "[control $year] eval skip, eval exists" > "$log"
  fi
}

run_parallel() {
  local kind="$1"; shift
  local years=("$@")
  local -a free=()
  local -A pid_gpu=()
  local -A pid_label=()
  local qi=0 done_n=0 fail_n=0

  for g in "${GPUS[@]}"; do
    for ((j=0; j<JOBS_PER_GPU; j++)); do free+=("$g"); done
  done

  echo "============================================================="
  echo "kind=$kind years=${years[*]}"
  echo "gpus=${GPUS[*]} jobs_per_gpu=$JOBS_PER_GPU"
  echo "============================================================="

  reap_one() {
    wait -n 2>/dev/null
    for pid in "${!pid_gpu[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid"; rc=$?
        free+=("${pid_gpu[$pid]}")
        label="${pid_label[$pid]}"
        unset "pid_gpu[$pid]" "pid_label[$pid]"
        if [[ "$rc" -eq 0 ]]; then
          done_n=$((done_n + 1)); st="OK"
        else
          fail_n=$((fail_n + 1)); st="FAIL"
        fi
        echo "[$((done_n + fail_n))/${#years[@]}] $label $st"
      fi
    done
  }

  while [[ "$qi" -lt "${#years[@]}" || "${#pid_gpu[@]}" -gt 0 ]]; do
    while [[ "${#free[@]}" -gt 0 && "$qi" -lt "${#years[@]}" ]]; do
      gpu="${free[0]}"; free=("${free[@]:1}")
      year="${years[$qi]}"; qi=$((qi + 1))
      if [[ "$kind" == "continuous" ]]; then
        run_continuous_year "$gpu" "$year" &
      else
        run_control_year "$gpu" "$year" &
      fi
      pid=$!
      pid_gpu[$pid]="$gpu"
      pid_label[$pid]="${kind}_${year}"
    done
    [[ "${#pid_gpu[@]}" -gt 0 ]] && reap_one
  done

  echo "${kind}: done=$done_n fail=$fail_n"
  [[ "$fail_n" -eq 0 ]]
}

case "${1:-}" in
  smoke)
    run_parallel continuous 1950 2000
    ;;
  main)
    run_parallel continuous 1931 1935 1939 1943 1947 1951 1955 1959 1963 1967 1971 1975 1979 1983 1987 1991 1995 1999 2003 2007 2011 2015 2019
    ;;
  control)
    run_parallel control 1931 1935 1939 1943 1947 1951 1955 1959 1963 1967 1971 1975 1979 1983 1987 1991 1995 1999 2003 2007 2011 2015 2019
    ;;
  years)
    shift
    if [[ "$#" -eq 0 ]]; then
      echo "usage: $0 years YEAR [YEAR ...]" >&2
      exit 2
    fi
    run_parallel continuous "$@"
    ;;
  *)
    echo "usage: $0 smoke|main|control|years YEAR [YEAR ...]" >&2
    exit 2
    ;;
esac
