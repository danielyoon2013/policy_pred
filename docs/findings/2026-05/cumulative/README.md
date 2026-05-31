# Round 1 — Cumulative LoRA chain + forced-distribution format SFT

**Date:** 2026-05 · **Status:** complete, archived baseline · **Reproduce:** `python analysis/round1_report.py`

## Question

Does a model trained only on text through year Y "believe in" a policy more strongly as Y
approaches the policy's enactment year E? We measure, for each policy, how the probe response
changes across the 90 year-stamped models as the lookback `Y − E` rises from −10 to 0.

## Method

- **Models:** a 90-year cumulative LoRA chain (1931–2020). Year-Y model = year-(Y−1) adapter +
  one more LoRA pass on year-Y synthetic text. Two backends: **Talkie 13B** and **nanochat 1.36B**.
- **Conditions (4):** `{talkie, nanochat} × {CPT-only, +SFT}`. The `+SFT` variants add a short
  format-teaching SFT pass on top (10k hand-templated pairs: 5k Yes/No balanced 50/50, 5k Likert
  **forced to 20%×5**).
- **Eval:** the variant policy battery — 196 US policies × ~100 paraphrase variants each, two probes
  per (policy, variant): a **Yes/No** probe (`mean_p_yes`) and a **5-point Likert** probe
  (`mean_score`). Coverage was lookback ∈ **[−10, 0]** only (the eval filter kept a policy for
  year-model Y iff `Y ≤ E ≤ Y+10`, so no post-enactment side).
- **Statistic:** per policy, OLS slope of the probe vs lookback over its [−10, 0] window; aggregate
  via a sign test (fraction of policies with positive slope) against a 50/50 null.

## Headline results (Table A)

| Condition | Probe | n(slope>0)/196 | frac | binom p | mean Δ (lb 0 − lb −10) |
|---|---|---:|---:|---:|---:|
| talkie_cpt | Yes/No | 84 | 0.43 | 0.98 (null) | −0.043 |
| talkie_sft | Yes/No | 83 | 0.42 | 0.98 (null) | −0.032 |
| **nanochat_cpt** | **Yes/No** | **157** | **0.80** | **<1e-4** | **+0.029** |
| nanochat_sft | Yes/No | 118 | 0.60 | 0.002 | +0.005 |
| *(all four)* | Likert | 65–87 | 0.33–0.44 | ≥0.94 (null/reversed) | ≈0 |

Full tables (A headline, B per-decade, C/D per-topic) in [tables/round1_tables.md](tables/round1_tables.md);
per-policy long-format data in [tables/round1_per_policy.csv](tables/round1_per_policy.csv).

## Three findings

1. **The lookback signal exists — but only for nanochat CPT-only.** 80% of policies show belief
   rising toward enactment (p<1e-4). Talkie shows nothing in either condition (43%, i.e. slightly
   *reversed*). The smaller backbone, trained CPT-only, is the one that surfaces the era-specific
   belief. See [figures/lookback.png](figures/lookback.png).

2. **The Likert probe is dead.** All four conditions sit at/under chance (33–44% positive) and the
   absolute `mean_score` deltas are ~0. Mechanically, ≥55% of probability mass parks in the
   "Uncertain" middle bucket regardless of policy/year, so `mean_score = Σ k·p_k` is dominated by the
   k=0 constant and per-policy variation washes out. The forced-uniform 20%×5 Likert SFT distribution
   is the suspected cause → motivates Round 2's corpus-grounded SFT. See [figures/likert.png](figures/likert.png).

3. **SFT-on-top lifts the level, flattens the signal.** Format-teaching SFT raises mean `p_yes` by
   ~+0.15 uniformly but *weakens* the per-policy lookback signal (nanochat 80% → 60% positive). It
   teaches *how* to answer without exposing more era-specific belief.

### Secondary: the 1940s–50s reversal

The nanochat-CPT signal is strong in every decade **except the 1940s–50s**, where it reverses
(1940s 23% positive, 1950s 21%). The reversing policies are overwhelmingly Cold-War-startup /
national-security acts (Truman Doctrine, Marshall Plan, NSC/CIA, Internal Security Act, INA).
By topic, domestic policy (health, criminal justice, environment, civil rights, welfare, labor) all
show 76–83% positive slopes while national-security is 59% and finance 65%. The reversal is robust
across all four conditions. See [figures/by_decade_normalized.png](figures/by_decade_normalized.png),
[figures/topics_normalized.png](figures/topics_normalized.png), [figures/topics_summary.png](figures/topics_summary.png).

## Honest caveats

- Lookback coverage is one-sided ([−10, 0]); we cannot yet see whether belief *peaks* at enactment
  and falls after (the n-shape) — Round 2 adds the symmetric window.
- The per-topic tagger is a coarse keyword matcher (~30% of policies fall to "other"); Round 2 uses
  the catalog's `domain` column.
- Cumulative chaining confounds "year-Y belief" with "all training since 1931"; Round 2's
  rolling-window isolates a recency band.

## Artifacts

Source eval archives (local, not in repo): `C:/tmp/chain_evals.tgz` (talkie CPT),
`chain_sft_evals.tgz` (talkie+SFT), `chain_cpt_nanochat_evals.tgz` (nanochat CPT),
`chain_sft_nanochat_evals.tgz` (nanochat+SFT). Talkie SFT adapters: `D:/hist_LLM/policy_pred/sft_adapters.tar`.
