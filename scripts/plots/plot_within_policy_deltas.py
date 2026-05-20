"""Within-policy delta histogram: do policies trend positive or negative
between T=-X and T=0 (enactment year)?

Per-policy approach that sidesteps the model's policy-agnostic baseline
bias (the reason raw likert clusters at ~-0.10 and raw P(yes) at ~0.20).
By looking at the SHIFT each policy makes between two relative-time
anchors, we cancel out the baseline:

    delta_P = score(P, T=0) - score(P, T=-X)

If the hypothesis is right ("model belief rises as enactment approaches"),
the histogram of deltas should skew positive. If the null is right,
deltas distribute symmetrically around zero. The script also reports a
sign test and one-sample t-test so you can read a p-value.

Inputs are the same per-year eval.json files used by
plot_lookback_trajectory.py — no pod time needed.

Usage:
    python scripts/plots/plot_within_policy_deltas.py \\
        --evals-glob 'C:/tmp/chain_evals/policy_*_naive/eval.json' \\
        --mode likert5 \\
        --start -10 --end 0 \\
        --out figures/within_policy_deltas_likert.png
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path


def calendar_year_from_path(path: Path) -> int | None:
    s = str(path).lower()
    if "base" in s:
        return 1930
    m = re.search(r"(19\d{2}|20\d{2})", s)
    return int(m.group(1)) if m else None


def extract_policy_score(p_rec: dict, mode: str) -> float | None:
    scores = p_rec.get("scores", {})
    if mode == "yes_no":
        yn = scores.get("yes_no", {})
        return yn.get("mean_p_yes", yn.get("p_yes"))
    if mode == "likert5":
        lk = scores.get("likert5", {})
        return lk.get("mean_score", lk.get("score"))
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evals-glob", required=True,
                   help="Shell glob for per-year eval.json files.")
    p.add_argument("--mode", choices=["yes_no", "likert5"], default="likert5")
    p.add_argument("--start", type=int, default=-10,
                   help="Earlier relative-year anchor (default -10).")
    p.add_argument("--end", type=int, default=0,
                   help="Later relative-year anchor (default 0 = enactment year).")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default=None)
    return p.parse_args()


def main() -> None:
    import matplotlib.pyplot as plt
    args = parse_args()

    eval_paths = sorted(Path(p) for p in glob.glob(os.path.expanduser(args.evals_glob)))
    if not eval_paths:
        raise SystemExit(f"no eval.json matched {args.evals_glob!r}")

    # Index: policy_id -> {relative_year: value}
    by_policy: dict[str, dict[int, float]] = defaultdict(dict)
    for path in eval_paths:
        cal_year = calendar_year_from_path(path)
        if cal_year is None:
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for p_rec in data["results"].get("policies", []):
            pid = p_rec["id"]
            impl = p_rec.get("implementation_year")
            if impl is None:
                continue
            rel = cal_year - int(impl)
            val = extract_policy_score(p_rec, args.mode)
            if val is None:
                continue
            by_policy[pid][rel] = val

    # Compute per-policy delta = value(end) - value(start). Skip any policy
    # missing either anchor.
    deltas: list[tuple[str, float]] = []
    skipped_missing = 0
    for pid, vals in by_policy.items():
        if args.start in vals and args.end in vals:
            deltas.append((pid, vals[args.end] - vals[args.start]))
        else:
            skipped_missing += 1

    if not deltas:
        raise SystemExit(
            f"no policies have both T={args.start} and T={args.end} data points. "
            f"Try a less extreme --start (e.g. -5) since policies near the chain "
            f"edges may not have a full 10-year lookback."
        )

    delta_vals = [d for _, d in deltas]
    n = len(delta_vals)
    mean_d = sum(delta_vals) / n
    var_d = sum((d - mean_d) ** 2 for d in delta_vals) / max(n - 1, 1)
    std_d = math.sqrt(var_d)
    sem_d = std_d / math.sqrt(n) if n > 1 else 0.0
    t_stat = mean_d / sem_d if sem_d > 0 else 0.0

    n_pos = sum(1 for d in delta_vals if d > 0)
    n_neg = sum(1 for d in delta_vals if d < 0)
    n_zero = n - n_pos - n_neg

    # Two-sided p-value approximation via normal CDF on |t|. Adequate at n~100+.
    def _phi(z):
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))
    p_two = 2 * (1 - _phi(abs(t_stat))) if abs(t_stat) > 0 else 1.0

    print(f"Mode:            {args.mode}")
    print(f"T anchors:       {args.start} -> {args.end}")
    print(f"Policies (n):    {n}  ({skipped_missing} skipped — missing either anchor)")
    print(f"Mean delta:      {mean_d:+.5f}")
    print(f"Std delta:       {std_d:.5f}")
    print(f"SEM:             {sem_d:.5f}")
    print(f"t-stat:          {t_stat:+.3f}   (two-sided p ~= {p_two:.4f})")
    print(f"Sign counts:     positive={n_pos}  negative={n_neg}  zero={n_zero}")
    print(f"  % positive:    {100*n_pos/n:.1f}%")

    # Sort top-10 movers each direction for context
    deltas.sort(key=lambda kv: kv[1], reverse=True)
    print()
    print("Top 5 positive movers:")
    for pid, d in deltas[:5]:
        print(f"  {pid:35s} {d:+.4f}")
    print("Top 5 negative movers:")
    for pid, d in deltas[-5:]:
        print(f"  {pid:35s} {d:+.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    n_bins = max(20, int(math.sqrt(n)) * 2)
    ax.hist(delta_vals, bins=n_bins, color="#4c72b0", edgecolor="white", alpha=0.85)
    ax.axvline(0.0, color="grey", linestyle="--", linewidth=1.0,
               label="no change")
    ax.axvline(mean_d, color="#c44e52", linestyle="-", linewidth=2.0,
               label=f"mean = {mean_d:+.4f}")

    metric_name = "Likert score" if args.mode == "likert5" else "P(yes)"
    ax.set_xlabel(f"Δ {metric_name}  ( T={args.end} − T={args.start} )")
    ax.set_ylabel("Number of policies")
    title = (args.title or
             f"Within-policy delta ({metric_name}): T={args.start} → T={args.end}\n"
             f"n={n}, mean={mean_d:+.4f}, t={t_stat:+.2f}, p~={p_two:.3f}, "
             f"{100*n_pos/n:.0f}% positive")
    ax.set_title(title)
    ax.legend(loc="best")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
