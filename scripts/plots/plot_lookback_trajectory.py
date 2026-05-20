"""Cross-policy relative-time trajectory plot.

For each policy, plot the model's belief over a RELATIVE time axis where
X=0 is the enactment year and X=-10 is ten years before enactment. This
lets us overlay trajectories for many policies regardless of their
absolute enactment year, and look for a common pre-enactment lead-up
pattern.

Input: a directory or set of eval.json files from policy_battery (or
policy_battery_variants) where each filename embeds the calendar year
of the year-model that produced it (e.g.
`policy_1942_naive/eval.json` → year=1942). Each eval.json may contain
scores for many policies (the multi-policy panel run); we pull all
policies from each file and stitch them into trajectories.

Usage:
    python scripts/plots/plot_lookback_trajectory.py \\
        --evals-glob '~/policy_pred_data/experiments/policy_*/eval.json' \\
        --mode likert5 \\
        --lookback 10 \\
        --out figures/cross_policy_lookback.png

Reads the policy's `implementation_year` from each eval.json record to
compute relative_year = calendar_year - implementation_year.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evals-glob", required=True,
                   help="Shell glob for eval.json files (will expand ~). "
                        "Each path's parent dir name should encode the year, e.g. "
                        "policy_1942_naive/eval.json or policy_1942_base/eval.json.")
    p.add_argument("--mode", choices=["yes_no", "likert5"], default="likert5",
                   help="Which probe mode to plot (default: likert5).")
    p.add_argument("--policies", nargs="+", default=None,
                   help="Restrict to these policy ids. Default: all policies present "
                        "in any eval.json.")
    p.add_argument("--lookback", type=int, default=10,
                   help="X-axis range: [-lookback, 0]. Default: 10.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output PNG path.")
    p.add_argument("--title", default=None)
    p.add_argument("--show-error-bars", action="store_true",
                   help="Show variant-std error bars (only meaningful for "
                        "policy_battery_variants eval.json files).")
    return p.parse_args()


def calendar_year_from_path(path: Path) -> int | None:
    """Extract 4-digit year from a path like '.../policy_1942_naive/eval.json'.

    Returns None for 'base' or unparseable paths (which we treat as the
    pre-Talkie cutoff year, conventionally 1930).
    """
    s = str(path).lower()
    if "base" in s:
        return 1930  # treat base as the implicit pre-1931 anchor
    m = re.search(r"(19\d{2}|20\d{2})", s)
    return int(m.group(1)) if m else None


def extract_policy_score(p_rec: dict, mode: str) -> tuple[float | None, float | None]:
    """Return (point_value, std) for one policy record under the chosen mode."""
    scores = p_rec.get("scores", {})
    if mode == "yes_no":
        yn = scores.get("yes_no", {})
        return (yn.get("mean_p_yes", yn.get("p_yes")), yn.get("std_p_yes"))
    elif mode == "likert5":
        lk = scores.get("likert5", {})
        return (lk.get("mean_score", lk.get("score")), lk.get("std_score"))
    return (None, None)


def main() -> None:
    import matplotlib.pyplot as plt
    args = parse_args()

    eval_paths = sorted(Path(p) for p in glob.glob(os.path.expanduser(args.evals_glob)))
    if not eval_paths:
        raise SystemExit(f"no eval.json matched {args.evals_glob!r}")

    # Build: policy_id -> list of (relative_year, value, std)
    by_policy: dict[str, list[tuple[int, float, float | None]]] = defaultdict(list)
    keep_ids = set(args.policies) if args.policies else None

    for path in eval_paths:
        cal_year = calendar_year_from_path(path)
        if cal_year is None:
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for p_rec in data["results"].get("policies", []):
            pid = p_rec["id"]
            if keep_ids is not None and pid not in keep_ids:
                continue
            impl = p_rec.get("implementation_year")
            if impl is None:
                continue
            rel = cal_year - int(impl)
            if rel < -args.lookback or rel > 0:
                continue
            val, std = extract_policy_score(p_rec, args.mode)
            if val is None:
                continue
            by_policy[pid].append((rel, val, std))

    if not by_policy:
        raise SystemExit("no policy data points survived the filter")

    print(f"Plotting {len(by_policy)} policies × up to {args.lookback + 1} year-points each")

    fig, ax = plt.subplots(figsize=(11, 6))

    # Per-policy lines (faint, individual trajectories)
    for pid, points in by_policy.items():
        points.sort()
        xs = [r for r, _, _ in points]
        ys = [v for _, v, _ in points]
        ax.plot(xs, ys, marker=".", linewidth=0.8, alpha=0.35, label=None)
        if args.show_error_bars:
            stds = [s if s is not None else 0.0 for _, _, s in points]
            ax.errorbar(xs, ys, yerr=stds, linewidth=0,
                        ecolor="grey", capsize=2, alpha=0.3)

    # Cross-policy mean trajectory (heavy, on top)
    rel_to_vals: dict[int, list[float]] = defaultdict(list)
    for points in by_policy.values():
        for r, v, _ in points:
            rel_to_vals[r].append(v)
    sorted_rels = sorted(rel_to_vals.keys())
    mean_xs = sorted_rels
    mean_ys = [sum(rel_to_vals[r]) / len(rel_to_vals[r]) for r in sorted_rels]
    ax.plot(mean_xs, mean_ys, marker="o", linewidth=3, color="#c44e52",
            label=f"Cross-policy mean ({args.mode})")

    # Reference lines + auto-scaled Y limits.
    # The theoretical range ([-1, +1] for likert, [0, 1] for P(yes)) is
    # usually much wider than what the model actually emits — likert mass
    # piles up near "Uncertain"=0 so observed values often live in [-0.25, 0].
    # Plotting at theoretical range hides the trajectory shape; instead pad
    # the actual data range by 20%.
    all_vals = [v for pts in by_policy.values() for _, v, _ in pts]
    lo_raw, hi_raw = min(all_vals), max(all_vals)
    span = hi_raw - lo_raw if hi_raw > lo_raw else 0.1
    pad = 0.2 * span
    y_lo, y_hi = lo_raw - pad, hi_raw + pad
    if args.mode == "likert5":
        ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="0 (uncertain / no signal)")
        ax.set_ylabel("Likert score (centered, [-1, +1])")
        # Keep 0 visible on the y-axis even if all data is below it — the
        # zero line is the methodological reference.
        ax.set_ylim(min(y_lo, -0.02), max(y_hi, 0.02))
    else:
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="0.5 (no signal)")
        ax.set_ylabel("P(yes)")
        ax.set_ylim(max(0.0, y_lo), min(1.0, y_hi))

    ax.axvline(0, color="red", linestyle=":", linewidth=1.0, alpha=0.7,
               label="enactment year")

    ax.set_xlabel("Years before enactment (0 = enactment year)")
    ax.set_xlim(-args.lookback - 0.5, 0.5)
    ax.set_xticks(list(range(-args.lookback, 1)))
    ax.set_title(args.title or
                 f"Cross-policy belief trajectory ({len(by_policy)} policies, mode={args.mode})")
    ax.legend(loc="best")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
