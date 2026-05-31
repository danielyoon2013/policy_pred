# policy_pred — current state (2026-05-31)

Read this first, then `CLAUDE.md` (durable rules), then `docs/findings/` (results trail) and the
active plan at `.claude/plans/our-focus-is-build-mutable-lantern.md`.

## Goal & hypothesis
Year-stamped LLM models, each knowing text only through year Y, probed for belief in historical
policies. Hypothesis: belief in policy P rises as Y approaches P's enactment year E (and — to test in
Round 2 — peaks at E and falls after: an n-shape). Methodology paper, not a product.

## Round 1 — COMPLETE (cumulative chain + forced-distribution format SFT)
Four conditions trained over a 90-year cumulative LoRA chain (1931–2020) and evaluated on the variant
policy battery (196 policies × ~100 paraphrases, Yes/No + Likert probes). Full writeup:
`docs/findings/2026-05/cumulative/` (reproduce with `python analysis/round1_report.py`).

Headline: **the lookback signal appears for nanochat CPT-only** (80% of policies, p<1e-4); Talkie is
flat in both conditions; the **Likert probe is dead** (collapses to ~85% "Uncertain"); a robust
**1940s–50s reversal** on national-security / Cold-War-startup policies. SFT-on-top lifts the response
level but flattens the per-policy signal.

Artifacts (local, not in repo): eval archives `C:/tmp/chain*_evals.tgz` (4 conditions); Talkie SFT
adapters `D:/hist_LLM/policy_pred/sft_adapters.tar`. The cumulative CPT adapters were deleted as the
SFT-on-top chain consumed them (eval.jsons preserved in the tarballs).

## Round 2 — IN PROGRESS (rolling-window, n-shape eval, per-year corpus SFT, scaling study)
Plan approved; see the plan file. Five methodology changes:
1. **Replace** cumulative chaining with **parallel rolling-window** training (each year-model = base +
   LoRA on a recency-weighted window; no `--init-from` chain). New module `windowing.py`.
2. **Per-year corpus-grounded SFT** (`synth/sft_corpus/`) — stance + distribution emerge from each
   year's legal corpus; no forced 20%×5. Aims to rescue the Likert probe.
3. **Symmetric eval window** (`window_years` in `policy_battery_variants.py`) → lookback ∈ [−10, +10]
   to see the n-shape.
4. **Scaling sweep** of per-year training volume {5k,10k,20k,40k}, sparse years equalized to 40k by
   identical resample-with-replacement (professor-decided; distinct-records-first ordering;
   `effective_unique_count` recorded per (year, level)).
5. **Eval matrix** {nanochat, talkie} × {LoRA, LoRA+SFT} × {5k…40k}, every cell a saved checkpoint.

Current sub-status: Phase 0 (findings log + this update) underway; Phases 1–4 pending.

## Current architecture (real, not stub)
- `backends/{base,talkie,nanochat,qwen}.py` + `backends/__init__.py:load_backend` — PEP-544 backend
  dispatch by `model_type`. Talkie 13B + nanochat 1.36B both real and validated (Phase 3 benchmarks:
  Talkie+LoRA-SFT 43.9% ARC / 58.6% RACE / 34.9% GSM-MC; nanochat 33.3/43.0/27.5; both beat the
  hist_llm full-FT reference using LoRA r=32).
- `train.py` — stateless LoRA/SFT trainer (`--data --experiment --init-from --sft --out`). CUDA only.
- `eval.py` + `evaluators/{policy_battery,policy_battery_variants,arc_mc,race_mc,gsm_mc,...}.py`.
- `synth/{naive,2step,sft_format}/` generators; `sources/{bulk_corpus,year_slice_legal}.py`.
- `scripts/pod/*.sh` runbooks; `scripts/data_prep/*.py` (synth batch, year YAMLs, variants).
- `config.py` path helpers; data root `D:/hist_LLM/policy_pred` (talkie weights 53 GB there).

## Data layout (on D:)
```
D:/hist_LLM/policy_pred/
  models/talkie_base/{final.ckpt, vocab.txt}
  years/{Y}/{legal.parquet, naive/synth.jsonl}
  experiments/<name>/{checkpoint/, eval.json}      # Round 2 adds experiments/windows/<combo>/
```
Source corpus: `D:/hist_LLM/corpus/raw/{year}/subset_*.parquet` (caselaw ≥200w/yr: median ~32.5k,
range 116–72.7k; 1940s–60s sparse). Bulk corpus spans 1678–2023.

## Catalog
`us_policy_event_battery_v4.csv` — 211 policy events (1930–2022), with a `domain` topic column and a
normative `benchmark_question` per event. Variant paraphrases in `data_artifacts/question_variants/`
(one .jsonl per event, ~100 each). Note: the older 5-policy `policies/catalog.yaml` is superseded by
the CSV battery for the variant evaluator.

## Open questions / next step
- Professor confirmed: equalize sparse years to 40k by resampling. Volume framing = per-year-X.
- Immediate next: finish Phase 0 (commit + push save point), then Phase 1 (regenerate 40k pools +
  build `synth/sft_corpus/`) and Phase 2 (`windowing.py` + symmetric eval) in parallel.
