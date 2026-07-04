# policy_pred — current state (2026-06-19)

Read this first, then `CLAUDE.md` (durable rules). **Findings now live in
`C:/Users/danielyoon/Dropbox/SWM/results/findings/` (moved out of docs/findings/).**

## Big picture: two SWM tracks (don't confuse them)
- **Policy track (THIS repo):** year-vintage LoRA models (1931–2020) probed on the 211-policy
  battery for P(yes | "does the US have policy X?") trajectories. Kaushik's battery
  (`us_policy_event_battery_v4.csv`) is the eval; this repo trains/evals the vintage models.
- **Values track (NOW BUILT IN THIS REPO — see `docs/VALUES_TRACK.md`):** survey-based benchmark from
  Kaushik's polling handoff (`SWM/kv_policy_events/.../2026-07-03_policy_benchmark_handoff`) —
  24-item pilot of multi-year GSS/Gallup items paired to policy events, scored against real survey
  marginals. Model probed **as a respondent** ("do you favor X"), p_focal trend across vintages vs the
  real trend + drift null. Battery `us_values_battery_v1.csv` + evaluator `values_battery` + runbook
  `scripts/pod/eval_values_battery.sh` + report `analysis/values_report.py` are all built and
  smoke-tested (stub end-to-end); AWAITING: user review of the 24 wordings, then a pod session
  (~30-45 GPU-min, both arms — can ride the deferred S2/reasoning session).

## Round 2 + SWM ablation — COMPLETE (2026-06). Five findings, all in SWM/results/findings/2026-06/:
1. **nanochat_sweep/** — LoRA-only ladder 5k/10k/20k × 1931–2020: pre-enactment rise exists but is
   ~80–90% calendar drift; CoT-SFT = level shift + domain homogenization, not trajectory destruction.
2. **caselaw_extended_eval/** — extended window rel −10..+20 (new `rel_min`/`rel_max` in
   `policy_battery_variants.py`): belief rises into enactment and PLATEAUS through +20 (no decay).
3. **talkie_sweep/** — talkie 13B vs nanochat 1.36B @10k have OPPOSITE slopes. Verified cause: base
   priors (talkie base says "No" to all 211 @0.13, unshiftable by rank-32 LoRA; nanochat base 0.72,
   malleable → tracks training era). Absolute "anticipation" is NOT robust across backbones.
4. **swm_lora_ablation/** — **the headline**: S1 (5k caselaw + 5k GST speeches @10k) vs caselaw @10k,
   matched volume, drift verified equal → **GST discourse DELAYS the belief rise** (pre-rise +0.30→+0.03,
   post-rise +0.25→+0.55). Discourse shifts belief from before to after enactment.
5. **reasoning_gate/** — yes_no vs yes_no_cot on 23 caselaw models: reasoning inflates P(yes) +0.40 on
   100% of cases (affirmation bias; model not trained to reason) → **direct yes_no stays the measure**.
   All 11,712 reasoning traces saved (`reasoning_traces.jsonl`). Caveat: 23-model/cap-8 subset → per-rel
   shape + per-topic panels are noisy.

## Durable archives (everything preserved; pods stopped)
```
D:/hist_LLM/policy_pred/
  round2_nanochat_1931_2020/           # caselaw adapters 5k/10k/20k (270) + evals
  round2_nanochat_1931_2020_cotsft/    # caselaw CoT-SFT adapters (90)
  round2_nanochat_caselaw_extended/    # extended-window (-10..+20) eval.jsons (90)
  round2_nanochat_swm/pp_swmgst/       # S1 caselaw+GST adapters (85) + evals
  round2_talkie_1931_2020/             # talkie adapters n5000+n10000 (180, 84GB) + n10000 evals
  reasoning_gate_evals/                # cot evals + reasoning_traces.jsonl
  base_probes/                         # talkie/nanochat base-only (no-LoRA) evals
  swm_seeds/{gst,economist,fomc}/      # ingested source passages (per-year parquet)
  synth_archive/years/<Y>/naive*/      # the OpenAI-generated synth (archived from C:/tmp)
```
Working copies (synth + pools, ~50GB) in `C:/tmp/policy_pred/` — pools rebuildable from synth.
nanochat base: `D:/hist_LLM/periods/1900_1949/model/`. talkie base: HF `talkie-lm/talkie-1930-13b-base`.

## Synth generation (what trained the models)
`scripts/data_prep/run_synth_batch.py` (OpenAI Batch, gpt-4o-mini) + prompts `synth/naive/prompts/`.
We used **synth_naive** (1 call/seed) — NOT the Anthropic-style two-stage `synth/2step/` (exists in
repo, unused; no batch-API version of it). Leakage: anachronistic policy-name docs filtered from
discourse blocks (S1/S2 at 0.000%; audit scripts were in C:/tmp).

## Deferred / next steps (need a fresh GPU pod, ~$150–200 total, unattended)
1. **S2 arm** (5k caselaw + 5k GST + 3k Econ/FOMC @13k, pools_swmall ready in C:/tmp + synth archived)
   — does more discourse amplify the delay effect? (+ caselaw-13k exact control.)
2. **Full-resolution reasoning sweep** (all 90 models; the 23-model subset was an unrequested
   downgrade — redo clean, ~5h).
3. Optional: naive-vs-2step synth A/B (diversity question); talkie n5000 eval; values-track integration.

## Known pitfalls (learned the hard way)
- Nanochat train cells need ~55GB GPU → **JOBS_PER_GPU=1** for training (2 OOMs); eval can run 2/GPU.
- Always set `NANOCHAT_WEIGHTS_DIR`/`POLICY_PRED_DATA_ROOT` when calling train/eval by hand.
- cot eval ≈33 min/model (cap-8); yes_no eval ≈2.4 min/model (cap-40).
- Archive pod weights BEFORE stopping pods. Check `n_distinct` in pool manifests (caselaw ceiling ~16.7k/yr).
