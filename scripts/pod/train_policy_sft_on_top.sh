#!/usr/bin/env bash
# Pod runbook: Phase 4 SFT-on-top of the cumulative CPT chain.
#
# Per year Y: start from the existing CPT-only adapter at
# policy_<Y>_naive/checkpoint and train ~250 SFT steps on
# data_links/sft/format_sft.jsonl (10K format-teaching pairs:
# 5K yes/no balanced + 5K likert balanced). LoRA rank is inherited
# from the CPT adapter via --init-from; --sft applies assistant-only
# loss masking.
#
# Output adapter lands at policy_<Y>_naive_sft/checkpoint via the
# auto-generated experiments/policy_<Y>_naive_sft.yaml (clone of the
# CPT YAML with name renamed).
#
# Disk-management: as soon as a year's SFT adapter is verified saved,
# the corresponding CPT-only adapter is deleted. We already have the
# CPT-only eval.json files locally in chain_evals.tgz, so no info loss.
#
# Usage:
#   bash scripts/pod/train_policy_sft_on_top.sh                          # talkie, all years
#   bash scripts/pod/train_policy_sft_on_top.sh 1935 1935                # talkie smoke
#   VARIANT=nanochat bash scripts/pod/train_policy_sft_on_top.sh         # nanochat chain
#   VARIANT=nanochat bash scripts/pod/train_policy_sft_on_top.sh 1935 1935  # nanochat smoke
#   KEEP_CPT=1 bash scripts/pod/train_policy_sft_on_top.sh ...           # don't delete CPT
#
# VARIANT picks which CPT chain to SFT-on-top. Empty = talkie (CPT path
# policy_<Y>_naive); "nanochat" = nanochat (CPT path policy_<Y>_naive_nanochat).
# SFT outputs land at policy_<Y>_naive[_${VARIANT}]_sft.
#
# Per-year wall time on 4x H100: ~5-7 min talkie / ~2-3 min nanochat.
# Full 90 years: ~10-11 hr talkie / ~3-5 hr nanochat. Idempotent.
#
set -uo pipefail   # NOT -e — let per-year failures pass so the rest of the chain proceeds

START_YEAR="${1:-1931}"
END_YEAR="${2:-2020}"
SFT_DATA="${SFT_DATA:-data_links/sft/format_sft.jsonl}"
KEEP_CPT="${KEEP_CPT:-0}"
VARIANT="${VARIANT:-}"                  # "" = talkie, "nanochat" = nanochat chain
SUFFIX="${VARIANT:+_${VARIANT}}"        # "" or "_nanochat"

NUM_PROCESSES="${NUM_PROCESSES:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [[ -z "$NUM_PROCESSES" || "$NUM_PROCESSES" -lt 1 ]]; then NUM_PROCESSES=4; fi

export TALKIE_WEIGHTS_DIR="${TALKIE_WEIGHTS_DIR:-${HOME}/talkie_base}"
export NANOCHAT_WEIGHTS_DIR="${NANOCHAT_WEIGHTS_DIR:-${HOME}/nanochat_base}"
export POLICY_PRED_DATA_ROOT="${POLICY_PRED_DATA_ROOT:-${HOME}/policy_pred_data}"

EXPS_DIR="${POLICY_PRED_DATA_ROOT}/experiments"

echo "============================================================="
echo "  Pod runbook: policy SFT-on-top chain"
echo "============================================================="
echo "  Year range:    ${START_YEAR}..${END_YEAR}"
echo "  Variant:       ${VARIANT:-talkie (default)}"
echo "  SFT data:      ${SFT_DATA}"
echo "  GPUs:          ${NUM_PROCESSES}"
echo "  Data root:     $POLICY_PRED_DATA_ROOT"
echo "  Delete CPT after SFT: $([[ "$KEEP_CPT" == "1" ]] && echo "no (KEEP_CPT=1)" || echo "yes")"
echo

if [[ -d ".venv" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# --- Sanity checks ---
if [[ ! -f "$SFT_DATA" ]]; then
    echo "ERROR: SFT data missing at $SFT_DATA"
    echo "       Upload synth/sft_format/out/sft_format.jsonl from local first."
    exit 1
fi
n_sft=$(wc -l < "$SFT_DATA")
echo "  SFT records: $n_sft (expect ~10000)"
echo

# --- Per-year loop ---
start_time=$(date +%s)
n_trained=0
n_skipped=0
n_failed=0
n_cpt_deleted=0

for y in $(seq "$START_YEAR" "$END_YEAR"); do
    base_name="policy_${y}_naive${SUFFIX}"
    sft_name="${base_name}_sft"
    cpt_dir="${EXPS_DIR}/${base_name}/checkpoint"
    sft_dir="${EXPS_DIR}/${sft_name}/checkpoint"
    cpt_yaml="experiments/${base_name}.yaml"
    sft_yaml="experiments/${sft_name}.yaml"

    echo "============================================================"
    echo "  YEAR $y  (SFT-on-top of ${base_name} CPT adapter)"
    echo "============================================================"

    # Idempotent skip.
    if [[ -f "$sft_dir/adapter_config.json" ]]; then
        echo "  SKIP: SFT adapter already at $sft_dir"
        n_skipped=$((n_skipped + 1))
        # If SFT is done but CPT lingers from a prior run, clean up now.
        if [[ "$KEEP_CPT" != "1" ]] && [[ -f "$cpt_dir/adapter_config.json" ]]; then
            echo "  cleanup: deleting stale CPT-only adapter at $cpt_dir"
            rm -rf "${EXPS_DIR}/${base_name}/checkpoint"
            n_cpt_deleted=$((n_cpt_deleted + 1))
        fi
        continue
    fi

    # CPT adapter must exist (--init-from source).
    if [[ ! -f "$cpt_dir/adapter_config.json" ]]; then
        echo "  ERROR: CPT-only adapter missing at $cpt_dir"
        echo "         SFT-on-top requires the ${base_name} CPT adapter."
        n_failed=$((n_failed + 1))
        continue
    fi

    # CPT YAML must exist (we clone it to make the SFT YAML).
    if [[ ! -f "$cpt_yaml" ]]; then
        echo "  ERROR: CPT YAML missing at $cpt_yaml"
        n_failed=$((n_failed + 1))
        continue
    fi

    # Generate the SFT YAML if missing — clone of the CPT YAML with
    # `name:` renamed so train.py/eval.py auto-route outputs to a
    # separate experiment directory. model_type (if present) carries over.
    if [[ ! -f "$sft_yaml" ]]; then
        echo "  generating $sft_yaml"
        python3 - <<PYEOF
from pathlib import Path
src = Path("${cpt_yaml}")
dst = Path("${sft_yaml}")
text = src.read_text(encoding="utf-8")
text = text.replace("name: ${base_name}", "name: ${sft_name}", 1)
dst.write_text(text, encoding="utf-8")
PYEOF
    fi

    # Stub-dir guard (crashed prior run could have left empty dir).
    if [[ -d "$sft_dir" ]]; then
        echo "  WARN: stub SFT dir at $sft_dir (no adapter_config.json); removing"
        rm -rf "$sft_dir"
    fi

    yr_start=$(date +%s)
    if accelerate launch --num_processes=$NUM_PROCESSES --mixed_precision=no train.py \
        --experiment "$sft_yaml" \
        --data "$SFT_DATA" \
        --init-from "$cpt_dir" \
        --batch-size 2 \
        --sft
    then
        yr_elapsed=$(( $(date +%s) - yr_start ))
        # Verify the new adapter actually landed.
        if [[ -f "$sft_dir/adapter_config.json" ]]; then
            echo "  SFT DONE: $((yr_elapsed / 60))m $((yr_elapsed % 60))s — adapter at $sft_dir"
            n_trained=$((n_trained + 1))

            # Disk: drop the CPT-only adapter now that the SFT version is saved.
            if [[ "$KEEP_CPT" != "1" ]]; then
                echo "  deleting CPT-only adapter at $cpt_dir to free disk"
                rm -rf "${EXPS_DIR}/${base_name}/checkpoint"
                n_cpt_deleted=$((n_cpt_deleted + 1))
            fi
        else
            echo "  ERROR: train.py exited OK but $sft_dir/adapter_config.json missing"
            n_failed=$((n_failed + 1))
        fi
    else
        echo "  TRAIN FAILED for year $y"
        n_failed=$((n_failed + 1))
    fi
done

total_elapsed=$(( $(date +%s) - start_time ))

echo
echo "============================================================="
echo "  SFT-on-top chain complete"
echo "============================================================="
echo "  Total wall time:    $((total_elapsed / 3600))h $((total_elapsed % 3600 / 60))m $((total_elapsed % 60))s"
echo "  SFT trained:        $n_trained"
echo "  Skipped (existed):  $n_skipped"
echo "  Failed:             $n_failed"
echo "  CPT adapters freed: $n_cpt_deleted"
echo
echo "Next: VARIANT=${VARIANT:-} bash scripts/pod/run_variants_eval_sft.sh"
