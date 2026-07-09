"""Consolidate every eval.json into ONE year x test leaderboard.

Scans an experiments dir for <name>/eval.json (written by eval.py), extracts a
single scalar per (model-year, test) across all four families, and writes to
reports/:
  leaderboard.csv          tidy long  : year, family, test, metric, value, n, experiment
  leaderboard_matrix.csv   wide        : rows = year, cols = test (+ lab_gap)
  leaderboard.md           the wide matrix as a markdown table (hand to the professor)

Per-family scalar:
  policy   -> mean P(yes) over the policy battery        (full N-shape: analysis/continuous_d23_rel20/)
  value    -> mean P(focal answer) over the survey items (trend vs surveys: analysis/values_report.py)
  lab      -> accuracy (+ derived lab_gap = acc - 0.25;  0 = no look-ahead bias)
  external -> accuracy per benchmark

Pure-Python (no pandas/numpy), matching the analysis/ conventions.

Usage:
  python analysis/leaderboard.py --experiments $HOME/pp_families/experiments
  python analysis/leaderboard.py --experiments <dir> --out reports
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

_YEAR = re.compile(r"(?:^|_)(\d{4})(?:_|$)")


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def summarize(ev: dict):
    """(family, test, metric_name, value, n) from one eval.json dict, or None."""
    evaluator = ev.get("evaluator")
    r = ev.get("results") or {}

    if evaluator == "nanochat_task":
        task = r.get("task", "task")
        family = "lab" if task == "lab" else "external"
        return family, task, "accuracy", r.get("accuracy"), r.get("n_total")

    if evaluator in ("arc_mc", "gsm_mc", "race_mc", "gsm8k"):
        test = evaluator[:-3] if evaluator.endswith("_mc") else evaluator
        val = r.get("accuracy") if r.get("accuracy") is not None else r.get("pass_at_1")
        return "external", test, "accuracy", val, r.get("n_total")

    if evaluator in ("policy_battery_variants", "policy_battery"):
        vals = []
        for p in (r.get("policies") or []):
            yn = (p.get("scores") or {}).get("yes_no") or {}
            vals.append(yn.get("mean_p_yes", yn.get("p_yes")))
        return "policy", "policy_belief", "mean_p_yes", _mean(vals), len(r.get("policies") or [])

    if evaluator == "values_battery":
        items = r.get("items") or []
        return "value", "value_pfocal", "mean_p_focal", _mean([it.get("p_focal") for it in items]), len(items)

    return None


def load_rows(exp_dir: Path) -> list[dict]:
    rows = []
    for ej in sorted(exp_dir.glob("*/eval.json")):
        try:
            ev = json.loads(ej.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {ej}: {e}")
            continue
        name = ev.get("experiment", ej.parent.name)
        m = _YEAR.search(name)
        s = summarize(ev)
        if s is None or m is None:
            continue
        family, test, metric, value, n = s
        rows.append({"year": int(m.group(1)), "family": family, "test": test,
                     "metric": metric, "value": value, "n": n, "experiment": name})
    return rows


# column order: policy, value, lab, then external benchmarks alphabetically
def _test_key(t: str):
    return ({"policy_belief": 0, "value_pfocal": 1, "lab": 2}.get(t, 3), t)


def write_leaderboard(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "leaderboard.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["year", "family", "test", "metric", "value", "n", "experiment"])
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["year"], _test_key(r["test"]))):
            w.writerow(r)

    years = sorted({r["year"] for r in rows})
    tests = sorted({r["test"] for r in rows}, key=_test_key)
    cell = {(r["year"], r["test"]): r["value"] for r in rows}
    has_lab = any(r["test"] == "lab" for r in rows)
    cols = list(tests) + (["lab_gap"] if has_lab else [])

    def gap(y):
        v = cell.get((y, "lab"))
        return v - 0.25 if isinstance(v, (int, float)) else None

    with open(out_dir / "leaderboard_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["year"] + cols)
        for y in years:
            line = [y] + [(f"{cell.get((y, t)):.4f}" if isinstance(cell.get((y, t)), (int, float)) else "")
                          for t in tests]
            if has_lab:
                g = gap(y)
                line.append(f"{g:+.4f}" if g is not None else "")
            w.writerow(line)

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

    md = ["# Eval leaderboard (year × test)\n",
          "| year | " + " | ".join(cols) + " |",
          "|---|" + "|".join(["---"] * len(cols)) + "|"]
    for y in years:
        line = [fmt(cell.get((y, t))) for t in tests]
        if has_lab:
            g = gap(y)
            line.append(f"{g:+.3f}" if g is not None else "—")
        md.append(f"| {y} | " + " | ".join(line) + " |")
    md.append(f"\n_{len(rows)} results across {len(years)} years. "
              "policy = mean P(yes); value = mean P(focal); lab/external = accuracy; "
              "lab_gap = acc − 0.25 (0 = no look-ahead bias)._")
    (out_dir / "leaderboard.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"wrote {out_dir}/leaderboard.csv + leaderboard_matrix.csv + leaderboard.md "
          f"({len(rows)} rows, {len(years)} years, cols={cols})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Consolidate eval.json -> year x test leaderboard")
    ap.add_argument("--experiments", required=True, type=Path, help="dir containing <name>/eval.json")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "reports")
    args = ap.parse_args()
    rows = load_rows(args.experiments)
    if not rows:
        print(f"no eval.json rows found under {args.experiments}")
        return
    write_leaderboard(rows, args.out)


if __name__ == "__main__":
    main()
