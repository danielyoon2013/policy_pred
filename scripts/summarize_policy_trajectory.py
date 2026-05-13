"""Summarize policy_battery eval results across a chain of year-models.

Each year's `python eval.py --experiment policy_<Y>_naive.yaml` already
saves a full eval.json. This script reads N of those, pulls out P(yes)
and likert score for the target policy, and prints/writes a tabular
summary so you can see the trajectory build up as the chain progresses.

Run repeatedly (e.g. after each year's eval finishes) to watch the
numbers evolve. Pair with plot_policy_trajectory.py for the figure.

Usage:
    # Glob form — picks up all completed evals:
    python scripts/summarize_policy_trajectory.py \\
        --policy social_security_1935 \\
        --glob '~/policy_pred_data/experiments/policy_*/eval.json'

    # Explicit form — guarantees order:
    python scripts/summarize_policy_trajectory.py \\
        --policy social_security_1935 \\
        --evals base.json y1931.json y1932.json ... \\
        --labels base 1931 1932 1933 1934 1935
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True,
                   help="Policy id to extract (e.g. social_security_1935).")
    p.add_argument("--glob", default=None,
                   help="Shell-style glob for eval.json files. Order is "
                        "auto-detected from year-like substrings in paths "
                        "(e.g. policy_1931_naive sorts before policy_1932_naive).")
    p.add_argument("--evals", nargs="+", type=Path, default=None,
                   help="Explicit list of eval.json paths (preserves order).")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Optional labels matching --evals 1:1 (e.g. base 1931 1932).")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional CSV output path. Stdout always prints.")
    return p.parse_args()


def auto_label(path: Path) -> str:
    """Pull a year-like or 'base' label from the experiment name.

    Check for 'base' BEFORE the year regex so that paths like
    policy_1931_base/eval.json get labeled 'base' rather than '1931'.
    """
    s = str(path).lower()
    if "base" in s:
        return "base"
    m = re.search(r"(\d{4})", s)
    if m:
        return m.group(1)
    return path.parent.name


def extract(path: Path, policy_id: str) -> dict | None:
    """Pull policy scores from an eval.json. Handles both:
      - policy_battery (single-question): scores.yes_no.p_yes, scores.likert5.score
      - policy_battery_variants (mean+std over N variants): scores.yes_no.mean_p_yes,
        std_p_yes, scores.likert5.mean_score, std_score
    Returns None if the policy isn't in this file.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for p in data["results"]["policies"]:
        if p["id"] != policy_id:
            continue
        scores = p["scores"]
        out: dict = {}
        if "yes_no" in scores:
            yn = scores["yes_no"]
            # Variant-aware fields take precedence; fall back to single-question.
            if "mean_p_yes" in yn:
                out["p_yes"] = yn["mean_p_yes"]
                out["p_yes_std"] = yn.get("std_p_yes")
                out["n_variants"] = yn.get("n_variants")
            else:
                out["p_yes"] = yn.get("p_yes")
                out["p_yes_std"] = None
                if "logprobs" in yn:
                    out["logp_yes"] = yn["logprobs"].get("Yes")
                    out["logp_no"] = yn["logprobs"].get("No")
        if "likert5" in scores:
            lk = scores["likert5"]
            if "mean_score" in lk:
                out["likert"] = lk["mean_score"]
                out["likert_std"] = lk.get("std_score")
                out["n_variants"] = lk.get("n_variants", out.get("n_variants"))
            else:
                out["likert"] = lk.get("score")
                out["likert_std"] = None
        return out
    return None


def main() -> None:
    args = parse_args()

    if args.glob and args.evals:
        raise SystemExit("pass either --glob or --evals, not both")
    if args.glob:
        # Expand ~ and shell glob.
        pattern = os.path.expanduser(args.glob)
        eval_paths = sorted(Path(p) for p in glob.glob(pattern))
        if not eval_paths:
            raise SystemExit(f"no eval.json files matched {pattern!r}")
        labels = [auto_label(p) for p in eval_paths]
    elif args.evals:
        eval_paths = list(args.evals)
        labels = list(args.labels) if args.labels else [auto_label(p) for p in eval_paths]
        if len(labels) != len(eval_paths):
            raise SystemExit("--labels must match --evals 1:1 if provided")
    else:
        raise SystemExit("pass either --glob or --evals")

    # Order: 'base' first (Talkie pre-1931), then year labels ascending.
    def sort_key(item):
        label = item[1]
        if label == "base":
            return (-1, "")
        try:
            return (int(label), "")
        except ValueError:
            return (0, label)
    pairs = sorted(zip(eval_paths, labels), key=sort_key)

    # Detect whether ANY eval has variant std fields → drives header format.
    any_variants = any(
        (extract(p, args.policy) or {}).get("p_yes_std") is not None
        or (extract(p, args.policy) or {}).get("likert_std") is not None
        for p, _ in pairs
    )

    print(f"Policy: {args.policy}")
    if any_variants:
        header = (f"  {'label':<10s}  {'P(yes)':>8s}  {'pY_std':>7s}"
                  f"  {'likert':>8s}  {'lk_std':>7s}  {'n_var':>5s}")
    else:
        header = (f"  {'label':<10s}  {'P(yes)':>8s}  {'likert':>8s}"
                  f"  {'logp_Yes':>10s}  {'logp_No':>10s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows: list[dict] = []
    for path, label in pairs:
        scores = extract(path, args.policy)
        if scores is None:
            print(f"  {label:<10s}  (policy {args.policy!r} not found in {path.name})")
            continue
        p_yes = scores.get("p_yes")
        p_yes_std = scores.get("p_yes_std")
        likert = scores.get("likert")
        likert_std = scores.get("likert_std")
        n_var = scores.get("n_variants")

        line = f"  {label:<10s}"
        line += f"  {p_yes:>+8.3f}" if p_yes is not None else f"  {'-':>8s}"
        if any_variants:
            line += f"  {p_yes_std:>7.3f}" if p_yes_std is not None else f"  {'-':>7s}"
            line += f"  {likert:>+8.3f}" if likert is not None else f"  {'-':>8s}"
            line += f"  {likert_std:>7.3f}" if likert_std is not None else f"  {'-':>7s}"
            line += f"  {n_var:>5d}" if n_var is not None else f"  {'-':>5s}"
        else:
            line += f"  {likert:>+8.3f}" if likert is not None else f"  {'-':>8s}"
            logp_y = scores.get("logp_yes")
            logp_n = scores.get("logp_no")
            line += f"  {logp_y:>10.3f}" if logp_y is not None else f"  {'-':>10s}"
            line += f"  {logp_n:>10.3f}" if logp_n is not None else f"  {'-':>10s}"
        print(line)

        rows.append({
            "label": label,
            "path": str(path),
            "p_yes": p_yes,
            "p_yes_std": p_yes_std,
            "likert": likert,
            "likert_std": likert_std,
            "n_variants": n_var,
        })

    if args.out is not None:
        import csv
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote CSV: {args.out}")


if __name__ == "__main__":
    main()
