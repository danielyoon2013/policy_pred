"""Compare base vs naive-trained vs 2step-trained on a policy probe.

Reads three eval.json files (one per model: base, naive, 2step) produced by
`policy_battery` evaluator runs, and produces a single bar chart with 6 bars:
3 models x 2 modes (yes_no P(yes) and likert5 weighted score).

Run from repo root:
    python scripts/plot_policy_compare.py \
        --policy social_security_1935 \
        --base    D:/.../experiments/policy_1931_base/eval.json \
        --naive   D:/.../experiments/policy_1931_naive/eval.json \
        --twostep D:/.../experiments/policy_1931_2step/eval.json \
        --out figures/ss1935_naive_vs_2step.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True,
                   help="Policy id to extract (e.g. social_security_1935).")
    p.add_argument("--base", type=Path, required=True,
                   help="eval.json from the base run (no adapter).")
    p.add_argument("--naive", type=Path, required=True,
                   help="eval.json from the naive-trained run.")
    p.add_argument("--twostep", type=Path, required=True,
                   help="eval.json from the 2step-trained run.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output PNG path.")
    p.add_argument("--title", default=None,
                   help="Optional plot title (default derived from --policy).")
    return p.parse_args()


def extract_scores(eval_json_path: Path, policy_id: str) -> dict:
    """Pull P(yes) and likert score for one policy out of an eval.json file."""
    with open(eval_json_path, encoding="utf-8") as f:
        data = json.load(f)
    for p in data["results"]["policies"]:
        if p["id"] == policy_id:
            return {
                "yes_no": p["scores"]["yes_no"]["p_yes"],
                "likert5": p["scores"]["likert5"]["score"],
            }
    raise KeyError(f"policy {policy_id!r} not found in {eval_json_path}")


def main() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    args = parse_args()

    bars = {
        "base":    extract_scores(args.base, args.policy),
        "naive":   extract_scores(args.naive, args.policy),
        "2step":   extract_scores(args.twostep, args.policy),
    }

    print(f"Policy: {args.policy}")
    for label, scores in bars.items():
        print(f"  {label:<6s}  P(yes)={scores['yes_no']:.3f}  "
              f"likert={scores['likert5']:.3f}")

    # Layout: 3 model groups along x; each group has yes_no + likert side-by-side.
    labels = list(bars.keys())
    yes_no_vals = [bars[l]["yes_no"] for l in labels]
    likert_vals = [bars[l]["likert5"] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, yes_no_vals, width, label="P(yes) — yes/no probe", color="#4c72b0")
    rects2 = ax.bar(x + width/2, likert_vals, width, label="Likert score (0-1)", color="#dd8452")

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("P(implemented)")
    ax.set_title(args.title or f"P({args.policy} implemented) — base vs naive vs 2step")
    ax.set_xticks(x)
    ax.set_xticklabels(["base\n(no adapter)", "naive-trained", "2step-trained"])
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
               label="0.5 (no signal)")
    ax.legend(loc="upper left")

    # Value labels on each bar
    for rect, val in zip(list(rects1) + list(rects2), yes_no_vals + likert_vals):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
