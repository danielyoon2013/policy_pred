"""Regenerate the Round-1 findings report (tables + figures) from the eval tarballs.

Round 1 = the cumulative-chain experiment: four conditions
(talkie/nanochat × CPT-only/+forced-SFT) trained over a 90-year cumulative LoRA
chain and evaluated on the variant policy battery. This script is the single
reproducible source for the tables and plots in
docs/findings/2026-05/cumulative/ — it reads only the four `chain*_evals.tgz`
archives, so the report can be rebuilt from scratch without the pod.

Why a script and not pasted numbers: the headline claims (nanochat-CPT carries
the lookback signal, the Likert collapse, the 1940s-50s reversal) all derive
from per-policy slope statistics over the lookback window; keeping the
computation in one place means the report and the paper never drift apart.

Usage:
    python analysis/round1_report.py
    python analysis/round1_report.py --tarball-dir C:/tmp --out-dir docs/findings/2026-05/cumulative
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import tarfile
from collections import defaultdict
from pathlib import Path

# Condition label -> tarball filename. Each archive holds 90 per-year eval.json
# files at policy_<Y>_naive[...]/eval.json.
CONDITIONS = {
    "talkie_cpt": "chain_evals.tgz",
    "talkie_sft": "chain_sft_evals.tgz",
    "nanochat_cpt": "chain_cpt_nanochat_evals.tgz",
    "nanochat_sft": "chain_sft_nanochat_evals.tgz",
}

# Coarse keyword tagger for the per-topic tables. The catalog also carries a
# `domain` column (preferred for Round 2); this keyword map reproduces the
# Round-1 exploratory slicing exactly so the historical numbers match.
TOPICS = {
    "civil_rights": ["civil rights", "discrimination", "voting", "race", "minority",
                     "gay", "lgbt", "segregat", "fair", "equal"],
    "labor_wages": ["labor", "wage", "union", "collective", "overtime",
                    "employment", "workplace", "safety"],
    "environment": ["environment", "pollution", "climate", "air", "water",
                    "emission", "energy", "fuel"],
    "social_welfare": ["social security", "medicare", "medicaid", "welfare",
                       "unemployment", "snap", "tanf", "housing", "pension"],
    "natl_security": ["military", "defense", "war", "security", "surveillance",
                      "spy", "foreign"],
    "finance_banking": ["bank", "finance", "tax", "currency", "monetary",
                        "wall street", "securit", "antitrust", "monopoly"],
    "health_medical": ["health", "drug", "fda", "medical", "aids", "tobacco"],
    "crim_justice": ["crime", "criminal", "sentenc", "prison", "death penalty",
                     "firearm", "gun"],
}

# Probe index into each policy's score record. 0 = Yes/No p_yes, 1 = Likert mean_score.
PROBES = [("yes_no", "mean_p_yes"), ("likert", "mean_score")]


def load_condition(tarball: Path) -> tuple[dict, dict]:
    """Read one condition's archive into {year: {policy_id: (p_yes, mscore, enactment)}}.

    Also returns {policy_id: description} so the topic tagger has text to match.
    """
    by_year: dict[int, dict] = {}
    descriptions: dict[str, str] = {}
    with tarfile.open(tarball) as tf:
        for m in tf:
            if not m.isfile():
                continue
            year = int(m.name.split("/")[0].split("_")[1])
            data = json.load(tf.extractfile(m))
            row: dict[str, tuple] = {}
            for p in data["results"]["policies"]:
                s = p["scores"]
                row[p["id"]] = (
                    s["yes_no"]["mean_p_yes"],
                    s["likert5"]["mean_score"],
                    p.get("implementation_year"),
                )
                descriptions.setdefault(p["id"], p.get("description", "") or "")
            by_year[year] = row
    return by_year, descriptions


def _slope(xs: list[float], ys: list[float]) -> float:
    """Ordinary-least-squares slope; 0 for a degenerate x-spread."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / den


def _binom_p(n_pos: int, n: int) -> float:
    """One-sided normal-approx p-value for n_pos positives under a 50/50 null."""
    if n == 0:
        return float("nan")
    z = (n_pos / n - 0.5) / math.sqrt(0.25 / n)
    return 0.5 * (1 + math.erf(-z / math.sqrt(2)))


def per_policy_slopes(by_year: dict, probe_idx: int) -> list[dict]:
    """Per-policy lookback slope over lb in [-10, 0] (Round-1 coverage).

    For each policy enacted in year E, fit the probe value against lookback
    (Y - E) using the year-models that evaluated it. Returns one row per policy
    with >= 5 usable lookback points.
    """
    enacted = {pid: tup[2]
               for pols in by_year.values()
               for pid, tup in pols.items() if tup[2] is not None}
    rows = []
    for pid, e in enacted.items():
        traj = []
        for lb in range(-10, 1):
            y = e + lb
            if y in by_year and pid in by_year[y]:
                traj.append((lb, by_year[y][pid][probe_idx]))
        if len(traj) >= 5:
            xs = [t[0] for t in traj]
            ys = [t[1] for t in traj]
            rows.append({"id": pid, "e": e, "slope": _slope(xs, ys),
                         "delta": ys[-1] - ys[0]})
    return rows


def _classify(desc: str) -> list[str]:
    d = desc.lower()
    return [t for t, kws in TOPICS.items() if any(k in d for k in kws)]


def write_tables(loaded: dict, descriptions: dict, out_dir: Path) -> None:
    """Emit the four Round-1 tables as one markdown file + a long-format CSV."""
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    md: list[str] = []
    md.append("# Round-1 lookback tables (regenerated by analysis/round1_report.py)\n")
    md.append("Per-policy slope of the probe over lookback (Y - E) for lb in [-10, 0].")
    md.append("binom_p = one-sided p vs a 50/50 null. mean_delta = probe(lb=0) - probe(lb=-10).\n")

    # Table A — headline per condition, both probes.
    md.append("## Table A - Headline per condition\n")
    md.append("| Condition | Probe | n>0 | frac | binom_p | mean_delta | mean_slope |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for cond, by_year in loaded.items():
        for idx, (probe, _) in enumerate(PROBES):
            rows = per_policy_slopes(by_year, idx)
            n = len(rows)
            npos = sum(1 for r in rows if r["slope"] > 0)
            md.append(f"| {cond} | {probe} | {npos} | {npos/n:.2f} | "
                      f"{_binom_p(npos, n):.4f} | "
                      f"{sum(r['delta'] for r in rows)/n:+.4f} | "
                      f"{sum(r['slope'] for r in rows)/n:+.5f} |")
    md.append("")

    # Table B — fraction positive by decade x condition x probe.
    md.append("## Table B - Fraction positive by decade x condition x probe\n")
    md.append("Cells = n_pos/n_total = fraction. **Bold** = >=0.75 (strong signal).\n")
    cols = [f"{c}-{p}" for p, _ in PROBES for c in loaded]
    md.append("| Decade | " + " | ".join(cols) + " |")
    md.append("|---|" + "|".join("---" for _ in cols) + "|")
    for era in range(1930, 2030, 10):
        cells = [f"{era}s"]
        for idx in range(len(PROBES)):
            for cond in loaded:
                rows = [r for r in per_policy_slopes(loaded[cond], idx)
                        if (r["e"] // 10) * 10 == era]
                if not rows:
                    cells.append("-")
                    continue
                n = len(rows)
                npos = sum(1 for r in rows if r["slope"] > 0)
                frac = npos / n
                txt = f"{npos}/{n}={frac:.2f}"
                cells.append(f"**{txt}**" if frac >= 0.75 else txt)
        md.append("| " + " | ".join(cells) + " |")
    md.append("")

    # Tables C/D — per topic, one per probe.
    for idx, (probe, _) in enumerate(PROBES):
        letter = "C" if idx == 0 else "D"
        per_cond = {c: per_policy_slopes(loaded[c], idx) for c in loaded}
        md.append(f"## Table {letter} - Per-topic, probe={probe} "
                  f"(n | frac_pos | mean_delta)\n")
        md.append("| Topic | " + " | ".join(loaded) + " |")
        md.append("|---|" + "|".join("---" for _ in loaded) + "|")
        for topic in TOPICS:
            cells = [topic]
            for cond in loaded:
                sub = [r for r in per_cond[cond]
                       if topic in _classify(descriptions.get(r["id"], ""))]
                if not sub:
                    cells.append("-")
                    continue
                n = len(sub)
                npos = sum(1 for r in sub if r["slope"] > 0)
                cells.append(f"n={n}, {npos/n:.2f}, "
                             f"{sum(r['delta'] for r in sub)/n:+.3f}")
            md.append("| " + " | ".join(cells) + " |")
        md.append("")

    (tables_dir / "round1_tables.md").write_text("\n".join(md), encoding="utf-8")

    # Long-format CSV: one row per (condition, probe, policy) for ad-hoc pivots.
    with open(tables_dir / "round1_per_policy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "probe", "policy_id", "enactment_year",
                    "topics", "slope", "endpoint_delta"])
        for cond, by_year in loaded.items():
            for idx, (probe, _) in enumerate(PROBES):
                for r in per_policy_slopes(by_year, idx):
                    topics = "|".join(_classify(descriptions.get(r["id"], "")))
                    w.writerow([cond, probe, r["id"], r["e"], topics,
                                f"{r['slope']:.6f}", f"{r['delta']:.6f}"])
    print(f"  tables -> {tables_dir}")


def write_figures(loaded: dict, descriptions: dict, out_dir: Path) -> None:
    """Regenerate the headline lookback + per-decade-normalized figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    style = {"talkie_cpt": ("C0", "-"), "talkie_sft": ("C0", "--"),
             "nanochat_cpt": ("C3", "-"), "nanochat_sft": ("C3", "--")}

    # Figure 1: cross-policy mean lookback trajectory (yes/no).
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, by_year in loaded.items():
        by_lb = defaultdict(list)
        for y, pols in by_year.items():
            for pid, (py, _, e) in pols.items():
                if e is not None and -10 <= y - e <= 0:
                    by_lb[y - e].append(py)
        xs = sorted(by_lb)
        color, ls = style[cond]
        ax.plot(xs, [st.mean(by_lb[lb]) for lb in xs], color=color, linestyle=ls,
                marker="o", linewidth=2, label=cond)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("lookback (Y - enactment)")
    ax.set_ylabel("mean p_yes")
    ax.set_title("Round 1: cross-policy lookback trajectory (Yes/No)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "lookback.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: per-decade baseline-normalized trajectory for nanochat CPT
    # (the condition that carries the signal).
    by_year = loaded["nanochat_cpt"]
    enacted = {pid: tup[2] for pols in by_year.values()
               for pid, tup in pols.items() if tup[2] is not None}
    decade_traj: dict = defaultdict(lambda: defaultdict(list))
    decade_stats: dict = defaultdict(list)
    for pid, e in enacted.items():
        traj = [(lb, by_year[e + lb][pid][0]) for lb in range(-10, 1)
                if (e + lb) in by_year and pid in by_year[e + lb]]
        if len(traj) < 5:
            continue
        ys = [t[1] for t in traj]
        mu = sum(ys) / len(ys)
        decade_stats[(e // 10) * 10].append(_slope([t[0] for t in traj], ys))
        for lb, v in traj:
            decade_traj[(e // 10) * 10][lb].append(v - mu)
    decades = list(range(1930, 2030, 10))
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=True, sharex=True)
    for ax, era in zip(axes.flatten(), decades):
        by_lb = decade_traj[era]
        if not by_lb:
            ax.set_title(f"{era}s  (no data)")
            continue
        xs = sorted(by_lb)
        means = np.array([st.mean(by_lb[lb]) for lb in xs])
        cis = np.array([1.96 * (st.stdev(by_lb[lb]) if len(by_lb[lb]) > 1 else 0)
                        / max(len(by_lb[lb]), 1) ** 0.5 for lb in xs])
        slopes = decade_stats[era]
        frac = sum(1 for s in slopes if s > 0) / len(slopes) if slopes else 0
        color = "C2" if frac >= 0.6 else "C3"
        ax.fill_between(xs, means - cis, means + cis, alpha=0.25, color=color)
        ax.plot(xs, means, color=color, marker="o", linewidth=2)
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.4)
        ax.set_title(f"{era}s  n={len(slopes)} frac>0={frac:.0%}", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("Round 1: per-decade baseline-normalized lookback (nanochat CPT, Yes/No)")
    fig.tight_layout()
    fig.savefig(fig_dir / "by_decade_normalized.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  figures -> {fig_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tarball-dir", type=Path, default=Path("C:/tmp"),
                   help="Directory holding the four chain*_evals.tgz archives.")
    p.add_argument("--out-dir", type=Path,
                   default=Path("docs/findings/2026-05/cumulative"),
                   help="Where tables/ and figures/ are written.")
    p.add_argument("--no-figures", action="store_true",
                   help="Skip matplotlib figure regeneration (tables only).")
    args = p.parse_args()

    loaded: dict = {}
    descriptions: dict = {}
    for cond, fname in CONDITIONS.items():
        path = args.tarball_dir / fname
        if not path.exists():
            raise SystemExit(f"missing tarball: {path}")
        loaded[cond], descs = load_condition(path)
        descriptions.update(descs)
        print(f"  loaded {cond}: {len(loaded[cond])} years")

    write_tables(loaded, descriptions, args.out_dir)
    if not args.no_figures:
        write_figures(loaded, descriptions, args.out_dir)
    print("done.")


if __name__ == "__main__":
    main()
