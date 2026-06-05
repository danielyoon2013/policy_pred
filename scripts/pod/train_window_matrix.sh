#!/usr/bin/env bash
# Pod runbook: train the rolling-window matrix in parallel, GPU-pinned.
#
# Replaces the Round-1 sequential --init-from chain. Every (year, backend, spec,
# X) cell is independent (trains from the frozen base, no chain), so we run one
# single-GPU train.py per GPU concurrently via a sliding job pool — no DDP, true
# parallelism. Checkpoints route by the YAML `name` to
# $POLICY_PRED_DATA_ROOT/experiments/<name>/checkpoint, so eval finds them by name.
#
# Two stages (run lora first, then sft):
#   stage=lora : assemble the window data + train a fresh LoRA from base.
#   stage=sft  : SFT-on-top of each lora adapter using that year's corpus SFT
#                (years/<Y>/sft_corpus.jsonl) via --init-from + --sft.
#
# Usage:
#   bash scripts/pod/train_window_matrix.sh <backend> <spec> <stage> [start_year] [end_year]
#   # examples (Phase 3a/3b staged rollout — nanochat, roll10, all X):
#   JOBS_PER_GPU=2 bash scripts/pod/train_window_matrix.sh nanochat roll10 lora
#   JOBS_PER_GPU=2 bash scripts/pod/train_window_matrix.sh nanochat roll10 sft
#
# Env knobs:
#   X_LEVELS      per-year record levels (default "5000 10000 20000 40000")
#   JOBS_PER_GPU  concurrent jobs per GPU (1 for talkie 13B, 2 for nanochat 1.36B)
#   POOL_DIR      per-year LoRA pools (default $HOME/data/pools/<Y>.jsonl)
#   BATCH_SIZE    per-rank batch (default 8)
#
# Idempotent: skips a cell whose adapter_config.json already exists.
set -uo pipefail   # NOT -e — one cell failing must not kill the wave

BACKEND="${1:?usage: <backend> <spec> <stage> [start_year] [end_year]}"
SPEC="${2:?missing spec (roll10|exp5|cum|single)}"
STAGE="${3:?missing stage (lora|sft)}"
START_YEAR="${4:-1931}"
END_YEAR="${5:-2019}"

X_LEVELS="${X_LEVELS:-5000 10000 20000 40000}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
BATCH_SIZE="${BATCH_SIZE:-8}"

export TALKIE_WEIGHTS_DIR="${TALKIE_WEIGHTS_DIR:-${HOME}/talkie_base}"
export NANOCHAT_WEIGHTS_DIR="${NANOCHAT_WEIGHTS_DIR:-${HOME}/nanochat_base}"
export POLICY_PRED_DATA_ROOT="${POLICY_PRED_DATA_ROOT:-${HOME}/policy_pred_data}"

POOL_DIR="${POOL_DIR:-${HOME}/data/pools}"
SFT_DIR="${SFT_DIR:-${POLICY_PRED_DATA_ROOT}/years}"   # years/<Y>/sft_corpus.jsonl
EXPS_DIR="${POLICY_PRED_DATA_ROOT}/experiments"
YAML_DIR="experiments/windows"
SCRATCH="${SCRATCH:-/tmp/window_assembled}"
LOG_DIR="${LOG_DIR:-/tmp/window_logs}"
mkdir -p "$SCRATCH" "$LOG_DIR" "$YAML_DIR"

if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# GPU list.
mapfile -t GPUS < <(nvidia-smi -L 2>/dev/null | sed -E 's/GPU ([0-9]+):.*/\1/')
if [[ ${#GPUS[@]} -eq 0 ]]; then GPUS=(0); fi

echo "============================================================="
echo "  Window matrix: backend=$BACKEND spec=$SPEC stage=$STAGE"
echo "============================================================="
echo "  Years:       ${START_YEAR}..${END_YEAR}"
echo "  X levels:    ${X_LEVELS}"
echo "  GPUs:        ${GPUS[*]}  (x${JOBS_PER_GPU} jobs/GPU)"
echo "  Pools:       ${POOL_DIR}"
echo

# Build the GPU slot pool (each GPU id repeated JOBS_PER_GPU times).
free=()
for g in "${GPUS[@]}"; do for ((j=0; j<JOBS_PER_GPU; j++)); do free+=("$g"); done; done

# Build the job queue as "year X" pairs.
queue=()
for y in $(seq "$START_YEAR" "$END_YEAR"); do
    for x in $X_LEVELS; do
        queue+=("$y $x")
    done
done
n_queue=${#queue[@]}
echo "  Cells to train: $n_queue"
echo

declare -A pid_gpu=()
declare -A pid_name=()
qi=0
n_done=0
n_fail=0
start_time=$(date +%s)

run_cell() {
    # $1=gpu $2=year $3=X — runs in the background.
    local gpu="$1" y="$2" x="$3"
    local sftflag="" name lora_name data yaml init log
    if [[ "$STAGE" == "sft" ]]; then sftflag="--sft"; fi
    name=$(python -m windowing name --year "$y" --backend "$BACKEND" --spec "$SPEC" \
            --per-year "$x" $sftflag)
    log="${LOG_DIR}/${name}.log"

    if [[ -f "${EXPS_DIR}/${name}/checkpoint/adapter_config.json" ]]; then
        echo "  SKIP $name (exists)"; return 0
    fi
    rm -rf "${EXPS_DIR}/${name}/checkpoint" 2>/dev/null

    yaml=$(python -m windowing yaml --year "$y" --backend "$BACKEND" --spec "$SPEC" \
            --per-year "$x" $sftflag --out-dir "$YAML_DIR")

    if [[ "$STAGE" == "lora" ]]; then
        # Assemble the rolling-window training data just-in-time (CPU).
        data="${SCRATCH}/${name}.jsonl"
        python -m windowing assemble --year "$y" --spec "$SPEC" --per-year "$x" \
            --pool-dir "$POOL_DIR" --out "$data" > "$log" 2>&1
        # LADDER mode: this cell is the max level (x); one flat-LR run dumps the
        # adapter at each LADDER level's fraction (lvl/x) into that level's dir,
        # so a single run produces every level. Also emit each level's eval YAML.
        local extra=""
        if [[ -n "${LADDER:-}" ]]; then
            local parts=() lvl frac lname
            for lvl in $LADDER; do
                frac=$(python -c "print(round($lvl/$x, 6))")
                lname=$(python -m windowing name --year "$y" --backend "$BACKEND" \
                         --spec "$SPEC" --per-year "$lvl")
                python -m windowing yaml --year "$y" --backend "$BACKEND" --spec "$SPEC" \
                    --per-year "$lvl" --out-dir "$YAML_DIR" >/dev/null
                parts+=("${frac}:${EXPS_DIR}/${lname}/checkpoint")
            done
            local joined; printf -v joined '%s,' "${parts[@]}"; joined="${joined%,}"
            extra="--flat-lr --ladder ${joined}"
        fi
        CUDA_VISIBLE_DEVICES="$gpu" python train.py --experiment "$yaml" \
            --data "$data" --batch-size "$BATCH_SIZE" $extra >> "$log" 2>&1
        local rc=$?
        rm -f "$data"   # scratch; manifest beside the yaml is the record
    else
        # SFT-on-top: init from the LoRA-only adapter for this (year, X).
        lora_name=$(python -m windowing name --year "$y" --backend "$BACKEND" \
                     --spec "$SPEC" --per-year "$x")
        init="${EXPS_DIR}/${lora_name}/checkpoint"
        data="${SFT_DIR}/${y}/sft_corpus.jsonl"
        if [[ ! -f "$init/adapter_config.json" ]]; then
            echo "  SKIP $name (no LoRA base at $init)" > "$log"; return 1
        fi
        if [[ ! -f "$data" ]]; then
            echo "  SKIP $name (no SFT data at $data)" > "$log"; return 1
        fi
        CUDA_VISIBLE_DEVICES="$gpu" python train.py --experiment "$yaml" \
            --data "$data" --init-from "$init" --sft --batch-size "$BATCH_SIZE" \
            > "$log" 2>&1
        local rc=$?
    fi
    [[ -f "${EXPS_DIR}/${name}/checkpoint/adapter_config.json" ]] && return 0 || return 1
}

reap() {
    # Wait for any job, return its GPU to the pool, log result.
    wait -n 2>/dev/null
    for pid in "${!pid_gpu[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid"; local rc=$?
            free+=("${pid_gpu[$pid]}")
            local nm="${pid_name[$pid]}"
            unset 'pid_gpu[$pid]' 'pid_name[$pid]'
            if [[ $rc -eq 0 ]]; then n_done=$((n_done+1)); st="OK"; else n_fail=$((n_fail+1)); st="FAIL"; fi
            local el=$(( $(date +%s) - start_time ))
            echo "  [$((n_done+n_fail))/$n_queue] $nm $st (${el}s)"
        fi
    done
}

# Dispatch loop: fill free GPU slots from the queue; reap as jobs finish.
while [[ $qi -lt $n_queue || ${#pid_gpu[@]} -gt 0 ]]; do
    while [[ ${#free[@]} -gt 0 && $qi -lt $n_queue ]]; do
        gpu="${free[0]}"; free=("${free[@]:1}")
        read -r y x <<< "${queue[$qi]}"; qi=$((qi+1))
        name=$(python -m windowing name --year "$y" --backend "$BACKEND" --spec "$SPEC" \
                --per-year "$x" $([[ "$STAGE" == "sft" ]] && echo --sft))
        run_cell "$gpu" "$y" "$x" &
        pid=$!; pid_gpu[$pid]="$gpu"; pid_name[$pid]="$name"
    done
    [[ ${#pid_gpu[@]} -gt 0 ]] && reap
done

total=$(( $(date +%s) - start_time ))
echo
echo "============================================================="
echo "  Matrix stage '$STAGE' complete: $n_done ok, $n_fail failed, ${total}s"
echo "============================================================="
echo "Next: bash scripts/pod/eval_window_matrix.sh $BACKEND $SPEC $STAGE"
