"""Plot multi-year P(policy implemented) trajectory.

Reads N eval.json files (one per year + base) produced by `policy_battery`
evaluator runs. Produces a line plot with two series (yes_no P(yes) and
likert5 weighted score) over training-data-cutoff year.

The methodological story this plot tells:
  base (1930) → year-1931 model → year-1932 model → ... → year-1935 model
  Each year-Y model is base + cumulative LoRA on synth(1931..Y).
  The probe asks "is it likely the US has <SS-1935 description>?" with NO
  year token in the prompt. We measure how that implicit-belief signal
  shifts as more pre-enactment text enters training.

If the methodology works, the trajectory rises monotonically as Y →
enactment year. Flat trajectory = no signal. Already-high at base = synth
leakage.

Run:
    python scripts/plot_policy_trajectory.py \\
        --policy social_security_1935 \\
        --years base 1931 1932 1933 1934 1935 \\
        --evals path/to/base/eval.json path/to/1931/eval.json ... \\
        --out figures/ss1935_trajectory.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True,
                   help="Policy id (e.g. social_security_1935).")
    p.add_argument("--years", nargs="+", required=True,
                   help="Year labels for the X axis, in order. Use 'base' "
                        "for the no-adapter reference. e.g. base 1931 1932 1933 1934 1935.")
    p.add_argument("--evals", nargs="+", required=True, type=Path,
                   help="One eval.json path per year, in the same order as --years.")
    p.add_argument("--enactment-year", type=int, default=1935,
                   help="X-position to draw the enactment vertical line.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output PNG path.")
    p.add_argument("--title", default=None,
                   help="Optional plot title.")
    return p.parse_args()


def extract_scores(eval_json_path: Path, policy_id: str) -> dict:
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

    args = parse_args()

    if len(args.years) != len(args.evals):
        raise SystemExit(
            f"--years has {len(args.years)} entries but --evals has "
            f"{len(args.evals)}; they must match 1:1 in order."
        )

    yes_no_vals: list[float] = []
    likert_vals: list[float] = []
    for year, path in zip(args.years, args.evals):
        scores = extract_scores(path, args.policy)
        yes_no_vals.append(scores["yes_no"])
        likert_vals.append(scores["likert5"])

    print(f"Policy: {args.policy}")
    print(f"  {'year':<8}  {'P(yes)':>8}  {'likert':>8}")
    for y, p_yes, l in zip(args.years, yes_no_vals, likert_vals):
        print(f"  {y:<8}  {p_yes:>8.3f}  {l:>8.3f}")

    # Plot. X axis is integer position 0..N-1 because "base" isn't a year.
    fig, ax = plt.subplots(figsize=(9, 5))
    x = list(range(len(args.years)))

    ax.plot(x, yes_no_vals, marker="o", linewidth=2, color="#4c72b0",
            label="P(yes) — yes/no probe")
    ax.plot(x, likert_vals, marker="s", linewidth=2, color="#dd8452",
            label="Likert weighted score (0-1)")

    # Reference lines.
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
               label="0.5 (no signal)")

    # Mark enactment year if it appears among the year labels (as int string).
    enact_str = str(args.enactment_year)
    if enact_str in args.years:
        enact_x = args.years.index(enact_str)
        ax.axvline(enact_x, color="red", linestyle=":", linewidth=1.0, alpha=0.7,
                   label=f"enacted (Aug {args.enactment_year})")

    ax.set_xticks(x)
    ax.set_xticklabels(args.years)
    ax.set_xlabel("Training-data cutoff year (cumulative)")
    ax.set_ylabel("P(implemented)")
    ax.set_ylim(0, 1.0)
    ax.set_title(args.title or
                 f"P({args.policy} implemented) trajectory — base + cumulative CPT")
    ax.legend(loc="upper left")

    # Annotate each point with its value for readability.
    for xi, p_yes in zip(x, yes_no_vals):
        ax.annotate(f"{p_yes:.2f}", xy=(xi, p_yes),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color="#4c72b0")
    for xi, l in zip(x, likert_vals):
        ax.annotate(f"{l:.2f}", xy=(xi, l),
                    xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=8, color="#dd8452")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
