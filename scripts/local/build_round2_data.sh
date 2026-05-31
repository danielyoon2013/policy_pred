#!/usr/bin/env bash
# Round-2 data prep: LoRA pools + per-year corpus SFT, in one runbook (local).
#
# Generation is local (OpenAI), NOT pod. Three stages, idempotent:
#   1. LoRA distinct synth via the OpenAI BATCH API (run_synth_batch.py) —
#      n_per_seed=1, capped at N_SEEDS seeds/year -> <corpus-root>/<Y>/naive/synth.jsonl.
#      (Batch = ~50% cheaper + async; submit -> poll -> parse in one call.)
#   2. Build uniform pools: windowing build-pool resamples each year's distinct
#      synth (with replacement, distinct-first) to N_SEEDS -> POOL_DIR/<Y>.jsonl.
#   3. Per-year corpus SFT (synth/sft_corpus) -> <corpus-root>/<Y>/sft_corpus.jsonl.
#      Currently synchronous (ThreadPoolExecutor); fine for the ~2.5k/yr volume.
#
# After this, stage POOL_DIR and the sft_corpus.jsonl files to the pod, then run
# scripts/pod/{train,eval}_window_matrix.sh.
#
# Usage:
#   bash scripts/local/build_round2_data.sh <start_year> <end_year> [stage]
#   # full default range, all stages:
#   bash scripts/local/build_round2_data.sh 1931 2019
#   # just the validation slice, just the LoRA stage:
#   bash scripts/local/build_round2_data.sh 1931 1945 lora
#   # stage in {lora|pool|sft|all} (default all)
#
# Env knobs:
#   N_SEEDS       seeds/year cap = uniform pool target (default 40000)
#   CORPUS_ROOT   per-year <Y>/{legal.parquet,naive/synth.jsonl} (default C:/tmp/policy_pred/years)
#   POOL_DIR      where build-pool writes <Y>.jsonl (default C:/tmp/policy_pred/pools)
#   SFT_N_SEEDS   passages/year for SFT (default from synth/sft_corpus/config.yaml = 2500)
#   KEY_FILE      OpenAI key file (default Dropbox/hist_LLM/key.txt)
#
# Cost (gpt-4o-mini, batch rate): LoRA ~$5/yr at 40k -> ~$450 for 90 yrs;
# SFT ~$0.30-0.60/yr -> ~$30-60. Wall: batch SLA up to 24h (usually 1-6h).
set -uo pipefail

START_YEAR="${1:?usage: <start_year> <end_year> [lora|pool|sft|all]}"
END_YEAR="${2:?missing end_year}"
STAGE="${3:-all}"

N_SEEDS="${N_SEEDS:-40000}"
CORPUS_ROOT="${CORPUS_ROOT:-C:/tmp/policy_pred/years}"
POOL_DIR="${POOL_DIR:-C:/tmp/policy_pred/pools}"
KEY_FILE="${KEY_FILE:-C:/Users/danielyoon/Dropbox/hist_LLM/key.txt}"
BATCH_WORK="${BATCH_WORK:-C:/tmp/policy_pred/batch_work_r2}"

if [[ ! -f "$KEY_FILE" ]]; then echo "ERROR: OpenAI key not at $KEY_FILE"; exit 1; fi
OPENAI_API_KEY=$(tr -d '[:space:]' < "$KEY_FILE"); export OPENAI_API_KEY
mkdir -p "$POOL_DIR" "$BATCH_WORK"

echo "============================================================="
echo "  Round-2 data prep: years ${START_YEAR}..${END_YEAR}  stage=${STAGE}"
echo "  N_SEEDS=${N_SEEDS}  corpus_root=${CORPUS_ROOT}  pool_dir=${POOL_DIR}"
echo "============================================================="

# --- Stage 1: LoRA distinct synth via batch API ---
if [[ "$STAGE" == "all" || "$STAGE" == "lora" ]]; then
    echo; echo ">>> Stage 1: LoRA synth (batch API, n_per_seed=1, n_seeds=${N_SEEDS})"
    python scripts/data_prep/run_synth_batch.py \
        --start-year "$START_YEAR" --end-year "$END_YEAR" \
        --n-seeds "$N_SEEDS" --n-per-seed 1 \
        --corpus-root "$CORPUS_ROOT" --batch-work-dir "$BATCH_WORK"
fi

# --- Stage 2: build uniform-40k pools (local, fast) ---
if [[ "$STAGE" == "all" || "$STAGE" == "pool" ]]; then
    echo; echo ">>> Stage 2: build pools -> ${POOL_DIR}/<Y>.jsonl"
    for y in $(seq "$START_YEAR" "$END_YEAR"); do
        distinct="${CORPUS_ROOT}/${y}/naive/synth.jsonl"
        if [[ ! -f "$distinct" ]]; then echo "  $y: SKIP (no synth at $distinct)"; continue; fi
        if [[ -f "${POOL_DIR}/${y}.jsonl" ]]; then echo "  $y: SKIP (pool exists)"; continue; fi
        python -m windowing build-pool --year "$y" --distinct "$distinct" \
            --out "${POOL_DIR}/${y}.jsonl" --target "$N_SEEDS"
    done
fi

# --- Stage 3: per-year corpus SFT ---
if [[ "$STAGE" == "all" || "$STAGE" == "sft" ]]; then
    echo; echo ">>> Stage 3: corpus SFT -> ${CORPUS_ROOT}/<Y>/sft_corpus.jsonl"
    sft_n_arg=""
    [[ -n "${SFT_N_SEEDS:-}" ]] && sft_n_arg="--n-seeds ${SFT_N_SEEDS}"
    python synth/sft_corpus/run.py --years "${START_YEAR}-${END_YEAR}" \
        --corpus-root "$CORPUS_ROOT" $sft_n_arg
fi

echo; echo "Done. Stage to pod:"
echo "  scp -r ${POOL_DIR}            pod:~/data/pools"
echo "  scp ${CORPUS_ROOT}/*/sft_corpus.jsonl  pod:~/policy_pred_data/years/<Y>/   (or rsync the tree)"
