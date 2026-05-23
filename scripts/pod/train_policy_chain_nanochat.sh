#!/usr/bin/env bash
# Pod runbook: build the 90-year nanochat CPT chain from scratch.
#
# Per year Y: train.py loads nanochat-base + (if Y>1931) the prior year's
# nanochat CPT adapter via --init-from, then continues pre-training on
# ~/data/policy_<Y>_naive.jsonl. Outputs land at
# policy_<Y>_naive_nanochat/checkpoint/ — separate from the talkie chain.
#
# This is the analogue of the existing Talkie chain (Phase 1-2) but for the
# nanochat backend, so we can compare backends head-to-head in Phase 5.
#
# Auto-generates experiments/policy_<Y>_naive_nanochat.yaml on the fly from
# the existing policy_<Y>_naive.yaml: rename `name:` and inject
# `model_type: nanochat`.
#
# Usage:
#   bash scripts/pod/train_policy_chain_nanochat.sh                # all years 1931..2020
#   bash scripts/pod/train_policy_chain_nanochat.sh 1935 1935      # smoke test
#   bash scripts/pod/train_policy_chain_nanochat.sh 1936 1945      # subrange
#
# Wall time (4x H100, batch_size=2 per rank, ~effective batch 8):
#   nanochat 1.36B is ~10x smaller than Talkie 13B → ~2-4 min/year
#   90 years → ~3-5 hr total. Idempotent (skip if checkpoint exists).
#
set -uo pipefail   # NOT -e — continue past per-year failures

START_YEAR="${1:-1931}"
END_YEAR="${2:-2020}"

NUM_PROCESSES="${NUM_PROCESSES:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [[ -z "$NUM_PROCESSES" || "$NUM_PROCESSES" -lt 1 ]]; then NUM_PROCESSES=4; fi

export NANOCHAT_WEIGHTS_DIR="${NANOCHAT_WEIGHTS_DIR:-${HOME}/nanochat_base}"
export POLICY_PRED_DATA_ROOT="${POLICY_PRED_DATA_ROOT:-${HOME}/policy_pred_data}"

DATA_DIR="${HOME}/data"
EXPS_DIR="${POLICY_PRED_DATA_ROOT}/experiments"

echo "============================================================="
echo "  Pod runbook: nanochat CPT chain training"
echo "============================================================="
echo "  Year range:     ${START_YEAR}..${END_YEAR}"
echo "  GPUs:           ${NUM_PROCESSES}"
echo "  Nanochat base:  $NANOCHAT_WEIGHTS_DIR"
echo "  Data root:      $POLICY_PRED_DATA_ROOT"
echo "  Synth files:    $DATA_DIR/policy_<Y>_naive.jsonl"
echo

# --- Sanity checks ---
if [[ ! -d "$NANOCHAT_WEIGHTS_DIR" ]]; then
    echo "ERROR: Nanochat weights missing at $NANOCHAT_WEIGHTS_DIR"
    exit 1
fi

if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# --- Verify prior chain link exists for non-1931 starts ---
if [[ "$START_YEAR" -ne 1931 ]]; then
    prev_ckpt="${EXPS_DIR}/policy_$((START_YEAR - 1))_naive_nanochat/checkpoint"
    if [[ ! -d "$prev_ckpt" ]]; then
        echo "ERROR: previous-year nanochat adapter missing at $prev_ckpt"
        echo "       Cumulative chain needs prior year. Start at 1931 or retrain earlier."
        exit 1
    fi
    echo "  Initial init-from: $prev_ckpt"
else
    echo "  Year 1931 is the first link in the chain → no --init-from (trains fresh on base)."
fi
echo

# --- Loop years ---
start_time=$(date +%s)
n_trained=0
n_skipped=0
n_failed=0

for y in $(seq "$START_YEAR" "$END_YEAR"); do
    base_name="policy_${y}_naive_nanochat"
    src_yaml="experiments/policy_${y}_naive.yaml"          # talkie YAML (template)
    dst_yaml="experiments/${base_name}.yaml"               # nanochat YAML (auto-gen)
    ckpt_dir="${EXPS_DIR}/${base_name}/checkpoint"
    synth_file="${DATA_DIR}/policy_${y}_naive.jsonl"
    prev_year=$((y - 1))
    prev_ckpt="${EXPS_DIR}/policy_${prev_year}_naive_nanochat/checkpoint"

    echo "============================================================"
    echo "  YEAR $y  (nanochat CPT, init-from year $prev_year)"
    echo "============================================================"

    # Idempotent skip.
    if [[ -f "$ckpt_dir/adapter_config.json" ]]; then
        echo "  SKIP: nanochat adapter already at $ckpt_dir"
        n_skipped=$((n_skipped + 1))
        continue
    fi
    if [[ -d "$ckpt_dir" ]]; then
        echo "  WARN: stub checkpoint dir at $ckpt_dir; removing"
        rm -rf "$ckpt_dir"
    fi

    # Source files.
    if [[ ! -f "$src_yaml" ]]; then
        echo "  ERROR: source YAML missing at $src_yaml"
        n_failed=$((n_failed + 1))
        continue
    fi
    if [[ ! -f "$synth_file" ]]; then
        echo "  ERROR: synth file missing at $synth_file"
        n_failed=$((n_failed + 1))
        continue
    fi
    if [[ "$y" -ne 1931 ]] && [[ ! -d "$prev_ckpt" ]]; then
        echo "  ERROR: prior-year nanochat adapter missing at $prev_ckpt"
        n_failed=$((n_failed + 1))
        continue
    fi

    # Auto-generate the nanochat YAML (rename name, inject model_type).
    if [[ ! -f "$dst_yaml" ]]; then
        echo "  generating $dst_yaml"
        python3 - <<PYEOF
from pathlib import Path
src = Path("${src_yaml}")
dst = Path("${dst_yaml}")
text = src.read_text(encoding="utf-8")
text = text.replace("name: policy_${y}_naive", "name: ${base_name}", 1)
if "model_type:" not in text:
    text = text.replace(
        "name: ${base_name}\n",
        "name: ${base_name}\nmodel_type: nanochat\n",
        1,
    )
dst.write_text(text, encoding="utf-8")
PYEOF
    fi

    n_records=$(wc -l < "$synth_file")
    echo "  Training on $synth_file ($n_records records)"

    if [[ "$y" -eq 1931 ]]; then
        init_from_args=""
        echo "  (no --init-from; year 1931 trains fresh on nanochat base)"
    else
        init_from_args="--init-from $prev_ckpt"
        echo "  --init-from $prev_ckpt"
    fi

    yr_start=$(date +%s)
    if accelerate launch --num_processes=$NUM_PROCESSES --mixed_precision=no train.py \
        --experiment "$dst_yaml" \
        --data "$synth_file" \
        --batch-size 2 \
        $init_from_args
    then
        yr_elapsed=$(( $(date +%s) - yr_start ))
        if [[ -f "$ckpt_dir/adapter_config.json" ]]; then
            echo "  TRAIN DONE: $((yr_elapsed / 60))m $((yr_elapsed % 60))s — $ckpt_dir"
            n_trained=$((n_trained + 1))
        else
            echo "  ERROR: train.py exited OK but $ckpt_dir/adapter_config.json missing"
            n_failed=$((n_failed + 1))
        fi
    else
        echo "  TRAIN FAILED for year $y. Stopping (cumulative chain can't skip a year)."
        n_failed=$((n_failed + 1))
        break
    fi
done

total_elapsed=$(( $(date +%s) - start_time ))

echo
echo "============================================================="
echo "  Nanochat CPT chain complete"
echo "============================================================="
echo "  Total wall time:  $((total_elapsed / 3600))h $((total_elapsed % 3600 / 60))m $((total_elapsed % 60))s"
echo "  Years trained:    $n_trained"
echo "  Years skipped:    $n_skipped"
echo "  Years failed:     $n_failed"
echo
echo "Next steps:"
echo "  VARIANT=nanochat bash scripts/pod/train_policy_sft_on_top.sh   # SFT-on-top"
echo "  VARIANT=nanochat bash scripts/pod/run_variants_eval_sft.sh     # variant eval"
