# Values benchmark v2 — vintage model vs human survey (point-in-time)

Compare each **year-vintage LLM** to the **US human survey** of that same year: pose every survey question
we have human data for, score the survey's own answer options, and measure how well the model matches the
humans. No focal-direction hand-labeling, no tiering — we compare the model's full answer distribution to the
humans'. This is a fresh, auto-collapsed benchmark (NOT the 24-item pilot); it reuses only the evaluator
`evaluators/value/survey.py` (registry `values_battery`) and its DK `JUNK` set.

## Three-phase workflow

### Phase 1 — build the clean HUMAN dataset (local, no GPU)  → the reviewable ground truth
```
python values_track/v2/build_human.py --min-years 2      # default; --min-years 1 = truly-all
```
Collapses Kaushik's handoff into one clean human answer distribution per (question, year):
US topline only, DK dropped, matrix/multi-response excluded, shares normalized. Writes to
`D:/hist_LLM/policy_pred/values_v2/human/`:
`questions.csv`, `human_answers.csv` (item_id,year,answer_option→human_share), `human_raw_all.parquet`
(all raw incl foreign/subgroups), `dropped_log.csv`, `build_manifest.json`.
**Review gate:** eyeball `human_answers.csv` before spending GPU. (min-years 2 → 2,669 questions;
GSS 939 · Gallup 913 · Pew 441 · PRRI 376. Kept ALL — demographic items included, they just score
uninformatively.)

### Phase 2 — model sweep (GPU, parallel across pods)
```
python values_track/v2/make_battery.py                   # questions.csv -> battery.csv (+ repo copy)
# on each pod (git pull brings values_track/v2/battery.csv + the evaluator):
BATTERY_CSV=values_track/v2/battery.csv START_YEAR=1972 END_YEAR=2000 \
    bash scripts/pod/eval_values_battery.sh caselaw $HOME/policy_pred_data/experiments
#   parallelize: shard the year range across pods, each arm (caselaw, swmgst) to its own pod.
# then pull every pod's results into the local archive + verify BEFORE teardown:
bash values_track/v2/pull_and_merge.sh caselaw <ssh-host> '$HOME/pp_values_caselaw' 1972-2000
```
`focal_answer` in the battery = the human-plurality option (only to satisfy the evaluator schema; the
comparison uses the full `p_per_option`). Each cell ≈ 2–3 min; 1943–2020 × 2 arms ≈ 6–7 GPU-hours (minutes
across a few pods).

### Phase 3 — compare + report (local, no GPU)
```
python values_track/v2/report.py \
    --evals caselaw=D:/hist_LLM/policy_pred/values_v2/eval/caselaw \
    --evals swmgst=D:/hist_LLM/policy_pred/values_v2/eval/swmgst
```
Joins model `p_per_option` to `human_answers.csv` on (item_id, year); writes `agreement_by_year.csv`,
`per_item.csv`, `figures/agreement_by_year.png`. Metrics: **argmax agreement** (model top == human plurality;
robust to LLM miscalibration) + **TV distance** (secondary).

## Local archive (everything lands here)
```
D:/hist_LLM/policy_pred/values_v2/{human/, battery/, eval/<arm>/, logs/<arm>/, report/, MANIFEST.md}
```
Join key: `item_id = slug(question_uid)` (unique by construction). Point-in-time only — no trend model, no CoT.
```
