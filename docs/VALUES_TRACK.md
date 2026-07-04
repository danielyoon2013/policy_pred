# Values track — survey-based benchmark for the year-vintage models

**Goal.** Test whether the year-vintage models hold the values of their era, graded against real
survey data. The policy track has no ground truth for public belief; the values track does: GSS/Gallup
asked Americans the same value questions for decades, so real percentages-by-year exist. Each vintage
model answers the survey questions *as a period respondent* (Kaushik/Leland's design decision); the
model's answer-probability trend across vintages is compared to the real survey trend. Deliverable:
one two-line chart per question + a trend correlation that must beat a calendar-drift null.

**Data provenance.** Kaushik's polling handoff
`SWM/kv_policy_events/policy_benchmark/handoffs/2026-07-03_policy_benchmark_handoff/normalized/`
(184k questions, 2.35M weighted answer rows). Pilot = 24 curated multi-year GSS/Gallup items paired to
policy events (`SWM/values_track/pilot/values_pilot_candidates.csv`).

## File map (mirrors the policy track piece-for-piece)

| role | file |
|---|---|
| battery (24 items: wording, options, focal answer) | `us_values_battery_v1.csv` (repo root) |
| ground truth (item × year × focal share) | `data_artifacts/values_truth_v1.csv` (gitignored, rebuildable) |
| builder (all judgment in one ITEM_SPECS dict) | `scripts/data_prep/build_values_battery.py` |
| evaluator (registered as `values_battery`) | `evaluators/values_battery.py` |
| pod runbook (per-arm sweep) | `scripts/pod/eval_values_battery.sh` |
| report (grid, correlations, drift null) | `analysis/values_report.py` |
| results | `docs/findings/2026-07/values_pilot/` |

## How to run

```bash
# 1. (Re)build battery + truth from the handoff; validates + spot-checks known series.
python scripts/data_prep/build_values_battery.py

# 2. On a pod (nanochat base + vintage adapters present, repo pulled):
bash scripts/pod/eval_values_battery.sh caselaw $HOME/policy_pred_data/experiments
bash scripts/pod/eval_values_battery.sh swmgst  $HOME/pp_swmgst/experiments
# ~30-45 GPU-min total for 90 years x 2 arms on 4 GPUs (24 items, scoring-only).

# 3. Pull /tmp/values_<arm>_evals.tgz locally, extract, then:
python analysis/values_report.py --evals caselaw=<dir> --evals swmgst=<dir>
```

## Design rules (locked; rationale in STATE.md findings)
- **Respondent framing** ("do you favor X"), the survey's own answer vocabulary, bare prompt.
- **Direct scoring only** — no CoT (reasoning inflates P(yes) +0.40 on 100% of cases: reasoning-gate).
- **nanochat vintages only** — talkie's 13B base is pinned at "No" (base-probe finding).
- **Trend is the statistic, not the level** (LLM probabilities are miscalibrated), and every matched
  correlation is reported against a **drift null** (same model series vs the other items' survey lines).
- **Exact-year scoring pairs**: each survey wave ≤2020 pairs with that same year's vintage; post-2020
  waves dropped; all 90 vintages drawn on plots.
- Arms: `caselaw` + `swmgst` (caselaw+GST) — the cross-track question "does discourse training improve
  value-tracking?". Scale past 24 items only after Leland/Kaushik review the pilot.
