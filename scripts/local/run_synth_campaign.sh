#!/usr/bin/env bash
# Run synth_naive across a YEAR RANGE with bounded parallelism.
#
# Why this script exists: the previous flow ran synth in batches of 4 jobs,
# waiting for each batch to finish before starting the next. That left ~95%
# of gpt-4o-mini's tier-2 rate-limit unused AND required manual chaining
# between batches. This script kicks off a sliding window of N parallel
# jobs that auto-refills as each finishes, so the campaign runs unattended
# at ~4x faster wall time.
#
# Usage:
#   bash scripts/local/run_synth_campaign.sh <start_year> <end_year> [parallelism] [n_seeds]
#
# Examples:
#   # Full 1950-2020 campaign at 16 parallel, 5k seeds each (~12 hr wall time):
#   bash scripts/local/run_synth_campaign.sh 1950 2020 16 5000
#
#   # Quick smoke test on 1955-1956 at 2 parallel, 100 seeds each:
#   bash scripts/local/run_synth_campaign.sh 1955 1956 2 100
#
# Defaults:
#   parallelism = 8  (32 concurrent OpenAI calls; safely under tier-2 limits)
#   n_seeds     = 5000
#
# Skips years whose synth.jsonl already exists (idempotent — safe to re-run
# after partial completion).
#
# Total OpenAI cost per year at 5k seeds: ~$5 (gpt-4o-mini)
#   1950-2020 (71 years) ≈ $350 total
#
# Wall time at 16 parallel × 8 workers, 71 years: ~12 hr
# Wall time at 8 parallel × 8 workers, 71 years:  ~22 hr
# Wall time at 4 parallel × 8 workers, 71 years:  ~45 hr  (the old slow path)
#
set -uo pipefail   # NOT -e — we want to continue on per-year failures

# --- Args ---
START_YEAR="${1:-1950}"
END_YEAR="${2:-2020}"
PARALLEL="${3:-8}"
N_SEEDS="${4:-5000}"
N_PER_SEED=4  # docs generated per seed call (synth_naive default)

# --- Paths ---
CORPUS_ROOT="C:/tmp/policy_pred/years"
LOG_ROOT="C:/tmp/policy_pred/synth_logs"
KEY_FILE="C:/Users/danielyoon/Dropbox/hist_LLM/key.txt"

mkdir -p "$LOG_ROOT"

# --- Env ---
if [[ ! -f "$KEY_FILE" ]]; then
    echo "ERROR: OpenAI key not found at $KEY_FILE"
    exit 1
fi
OPENAI_API_KEY=$(tr -d '[:space:]' < "$KEY_FILE")
export OPENAI_API_KEY

echo "============================================================="
echo "  synth_naive campaign: years ${START_YEAR}..${END_YEAR}"
echo "============================================================="
echo "  Parallelism: ${PARALLEL} concurrent jobs"
echo "  Per job:     ${N_SEEDS} seeds × ${N_PER_SEED} docs/seed"
echo "  Concurrent OpenAI calls: ~$((PARALLEL * 8)) (well under tier-2 limit)"
echo "  Logs:        ${LOG_ROOT}/synth_<year>.log"
echo

# --- Build the queue ---
queue=()
for y in $(seq "$START_YEAR" "$END_YEAR"); do
    out_file="${CORPUS_ROOT}/${y}/naive/synth.jsonl"
    parquet="${CORPUS_ROOT}/${y}/legal.parquet"
    if [[ -f "$out_file" ]]; then
        n=$(wc -l < "$out_file")
        echo "  $y: SKIP (already have $n records)"
        continue
    fi
    if [[ ! -f "$parquet" ]]; then
        echo "  $y: SKIP (no corpus parquet)"
        continue
    fi
    queue+=("$y")
done

n_queue=${#queue[@]}
echo
echo "  Total years to run: $n_queue"
if [[ $n_queue -eq 0 ]]; then
    echo "  Nothing to do."
    exit 0
fi
echo

# --- Sliding-window parallel runner ---
echo "============================================================="
echo "  Starting campaign"
echo "============================================================="

start_time=$(date +%s)
n_started=0
n_done=0

# Track pids -> year so we can log completions.
declare -A pid_to_year=()

launch_year() {
    local y=$1
    python synth_naive/run.py \
        --seeds "${CORPUS_ROOT}/${y}/legal.parquet" \
        --out   "${CORPUS_ROOT}/${y}/naive/" \
        --limit "$N_SEEDS" \
        --n-per-seed "$N_PER_SEED" \
        > "${LOG_ROOT}/synth_${y}.log" 2>&1 &
    local pid=$!
    pid_to_year[$pid]=$y
    n_started=$((n_started + 1))
    echo "  [$n_started/$n_queue] launched $y (pid=$pid)"
}

# Kick off the first wave.
for y in "${queue[@]:0:$PARALLEL}"; do
    launch_year "$y"
done

# Then as each finishes, launch the next year (sliding window).
remaining=("${queue[@]:$PARALLEL}")
while [[ ${#pid_to_year[@]} -gt 0 ]]; do
    # Wait for ANY one to finish.
    wait -n 2>/dev/null
    # Reap whichever pids are done.
    for pid in "${!pid_to_year[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            y=${pid_to_year[$pid]}
            unset 'pid_to_year[$pid]'
            n_done=$((n_done + 1))
            out_file="${CORPUS_ROOT}/${y}/naive/synth.jsonl"
            if [[ -f "$out_file" ]]; then
                n=$(wc -l < "$out_file")
                elapsed=$(( $(date +%s) - start_time ))
                echo "  [$n_done/$n_queue] $y DONE: $n records  (elapsed ${elapsed}s = $((elapsed / 60))min)"
            else
                echo "  [$n_done/$n_queue] $y FAILED: no synth.jsonl"
            fi
            # Launch next year if any remain.
            if [[ ${#remaining[@]} -gt 0 ]]; then
                next_y=${remaining[0]}
                remaining=("${remaining[@]:1}")
                launch_year "$next_y"
            fi
        fi
    done
done

total_elapsed=$(( $(date +%s) - start_time ))
echo
echo "============================================================="
echo "  Campaign complete"
echo "============================================================="
echo "  Total wall time: ${total_elapsed}s = $((total_elapsed / 60))min = $((total_elapsed / 3600))h"
echo "  Years completed: $n_done / $n_queue"
echo
echo "Verify with:"
echo "  for y in \$(seq $START_YEAR $END_YEAR); do"
echo "    n=\$(wc -l < ${CORPUS_ROOT}/\${y}/naive/synth.jsonl 2>/dev/null || echo MISSING)"
echo "    echo \"  \$y: \$n records\""
echo "  done"
