# nanochat LoRA-only ladder sweep — 1931–2020 (Round-2 core run)

**Date:** 2026-06. **Backend:** nanochat (1.36B, base = `hist_1900_1949` d23). **Method:** rolling
window `roll10` (backward 10-yr, no future leakage), per-year LoRA at three data levels via the
**save-as-you-go ladder** (one 20k-per-year run, flat-LR, adapters dumped at the 5k/10k/20k marks).
90 year-models × 3 levels = **270 adapters**, each evaluated on the **211-policy battery**
(`window_years=10`, so each policy is scored by year-models E−10…E+10). 12,186 measurements.

Probe = length-normalized **P(yes)** that "the United States has [policy]". Likert is dead
(collapses to Uncertain everywhere) — P(yes) is the working probe.

## Headline result — null-to-fragile (no robust anticipation)

Raw P(yes) drifts up gently across the ±10 window (~0.40→0.47) and stays near coin-flip throughout —
the model never strongly "believes" a policy is implemented, even post-enactment. Adversarial
verification showed the apparent **pre-enactment rise is ~80–90% an artifact** of:

1. **Calendar drift** — later year-models rate *every* policy higher (X=5k: mean P(yes) 0.36 in the
   1930s → 0.66 in the 2010s). Since within a policy `rel_year = year_model − const`, this secular
   model-era trend is mechanically confounded with "years-to-enactment." Detrending it collapses the
   per-policy pre-enactment slope from +0.0090 → **+0.0010 P(yes)/yr** (89% gone).
2. **New-Deal left-truncation** — 1930s policies supply 73% of the raw slope-sum from 11% of policies,
   and are truncated (year-models start 1931 → only ~4.6 pre-enactment points).

What survives: a fragile **+0.01/decade at X=10k only** (null at 5k/20k, fails a shuffled-enactment
placebo p=0.11). **X-scaling is non-monotonic** (5k=0.42, 10k=0.39, 20k=0.47 overall P(yes)) — more
data does not sharpen the trajectory. → Report as a **clean negative + a methodological fix**
(calendar-drift detrending must be the default analysis).

## Per-policy z-normalization (the right lens)

Each policy's P(yes) is centered + scaled by its own mean/std over its ±10-yr window, so the *shape*
shows up despite different baselines. See [figures/nshape_znorm.png](figures/nshape_znorm.png):
the raw-z curve ramps clearly, but the **calendar-detrended** panel shows the **pre-enactment portion
is nearly flat — most of the rise is post-enactment** ("policy now exists"), not anticipation.

## Domain breakdown (all 211 policies, 10 meaningful categories)

The 140 fragmented `domain` labels were collapsed to a curated 10-category taxonomy (first-token →
super-category; see `make_figures.py:TAXONOMY`), covering **every** policy:
Civil Rights 41 · Democracy & Governance 31 · Economy & Finance 29 · Civil Liberties 24 ·
Criminal Justice & Guns 19 · Natl Security & Foreign 17 · Health/Welfare/Educ 16 ·
Environment & Energy 15 · Labor 11 · Immigration 8  (= 211).

**Detrended pre-enactment z-rise (rel −10→0, X=10k)** — [figures/domain_increment_bar.png](figures/domain_increment_bar.png),
trajectories in [figures/domain_znorm_smallmult.png](figures/domain_znorm_smallmult.png):

| domain | n | detrended pre-rise |
|---|---|---|
| Natl Security & Foreign | 17 | **+0.85 z** |
| Civil Rights | 41 | **+0.44 z** |
| Criminal Justice & Guns | 19 | +0.29 z |
| Democracy & Governance | 31 | +0.22 z |
| Labor | 11 | +0.16 z |
| Health/Welfare/Educ | 16 | +0.08 z |
| Civil Liberties | 24 | +0.01 z |
| Environment & Energy | 15 | +0.01 z |
| Economy & Finance | 29 | −0.09 z |
| Immigration | 8 | **−0.47 z** |

**Even after detrending, the signal is heterogeneous by domain:** National Security/Foreign and Civil
Rights show a real pre-enactment rise; Immigration and Economy & Finance trend *down* approaching
enactment. The "weak overall" headline is an average of anticipating and anti-anticipating domains.
(Descriptive z-increments, not per-category significance-tested; some n are small — Immigration 8,
Labor 11.)

## Files
- `nano_tidy.csv` — 12,186 rows (X, year_model, policy_id, enactment_year, rel_year, p_yes, likert, domain, institution); reproduces all analysis.
- `make_figures.py` — taxonomy + z-normalization + detrending + the 3 figures.
- `figures/{nshape_znorm,domain_znorm_smallmult,domain_increment_bar}.png`
- Raw 270 `eval.json` live on pod1 + `C:/tmp/nano_results/nano_evals.tgz` (176 MB) — **pull before pod teardown** if the per-variant detail is wanted.

## Status / next
talkie (13B) ladder still training on pod2 → its eval is the decisive test: if the same
calendar-drift + domain pattern appears, the effect is method-wide (report negative); if the
detrended rise holds across all X levels in the bigger model, that is the first real evidence.
