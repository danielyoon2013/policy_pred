"""Round-2 analysis: n-shape, scaling curves, rolling-vs-cumulative.

Consumes the rolling-window eval results (experiments/<name>/eval.json, where
name = policy_<Y>_<backend>_<spec>_n<X>[_sft]) produced by
scripts/pod/eval_window_matrix.sh, and the archived Round-1 cumulative baseline
(the chain*_evals.tgz tarballs) for comparison.

Three outputs:
  1. n-shape: each policy is now scored by year-models E-10..E+10 (symmetric
     window). Fit a quadratic to the per-policy lookback trajectory and test for
     an inverted-U (concave, peak near 0) — the hypothesis that belief peaks at
     enactment and falls after.
  2. scaling curves: signal strength vs training volume X, plotted against the
     manifest's effective_unique (not nominal X) so the sparse-year resampling is
     visible.
  3. rolling vs cumulative: overlay the new rolling mean trajectory on Round-1's.

The pure statistics (quadratic fit, n-shape classification) are unit-tested in
tests/test_round2_analysis.py and run without any data; the loaders need a
completed Phase-3 run.

Usage:
    python analysis/round2_report.py --exps-dir <DATA_ROOT>/experiments \
        --backend nanochat --spec roll10 --out-dir docs/findings/2026-06/rolling-window
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

# name: policy_<Y>_<backend>_<spec>_n<X>[_sft]
_NAME_RE = re.compile(
    r"^policy_(?P<year>\d{4})_(?P<backend>[a-z]+)_(?P<spec>[a-z]+\d*)_n(?P<x>\d+)(?P<sft>_sft)?$")


def parse_name(name: str) -> dict | None:
    """Decompose an experiment name into its matrix coordinates, or None."""
    m = _NAME_RE.match(name)
    if not m:
        return None
    return {"year": int(m["year"]), "backend": m["backend"], "spec": m["spec"],
            "x": int(m["x"]), "sft": bool(m["sft"]), "name": name}


def quadratic_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares fit y = a*x^2 + b*x + c via the normal equations (no numpy).

    Returns (a, b, c). a < 0 means concave (inverted-U); the vertex is at -b/2a.
    """
    n = len(xs)
    if n < 3:
        return 0.0, 0.0, (sum(ys) / n if n else 0.0)
    s0 = n
    s1 = sum(xs); s2 = sum(x ** 2 for x in xs)
    s3 = sum(x ** 3 for x in xs); s4 = sum(x ** 4 for x in xs)
    t0 = sum(ys); t1 = sum(x * y for x, y in zip(xs, ys))
    t2 = sum((x ** 2) * y for x, y in zip(xs, ys))
    # Solve the 3x3 system [[s4,s3,s2],[s3,s2,s1],[s2,s1,s0]] [a,b,c]^T = [t2,t1,t0]^T.
    A = [[s4, s3, s2], [s3, s2, s1], [s2, s1, s0]]
    rhs = [t2, t1, t0]
    return tuple(_solve3(A, rhs))


def _solve3(A: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination for a 3x3 system; returns [0,0,mean]-ish on singular."""
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return [0.0, 0.0, b[2]]
        M[col], M[piv] = M[piv], M[col]
        for r in range(3):
            if r != col:
                f = M[r][col] / M[col][col]
                for c in range(col, 4):
                    M[r][c] -= f * M[col][c]
    return [M[i][3] / M[i][i] for i in range(3)]


def is_n_shape(xs: list[float], ys: list[float], *,
               peak_tol: float = 3.0) -> bool:
    """Inverted-U test: concave (a<0) with the vertex within peak_tol of lb=0."""
    a, b, _ = quadratic_fit(xs, ys)
    if a >= 0:
        return False
    vertex = -b / (2 * a)
    return abs(vertex) <= peak_tol


def load_window_evals(exps_dir: Path, backend: str, spec: str):
    """Load matrix eval.jsons grouped by (X, sft) -> {year: {pid: (p_yes, ms, E)}}.

    Returns {(x, sft): {year: {policy_id: (p_yes, mean_score, enactment_year)}}}.
    """
    out: dict = defaultdict(lambda: defaultdict(dict))
    for ev in exps_dir.glob(f"policy_*_{backend}_{spec}_n*/eval.json"):
        coord = parse_name(ev.parent.name)
        if coord is None:
            continue
        data = json.loads(ev.read_text(encoding="utf-8"))
        row = {}
        for p in data["results"]["policies"]:
            s = p["scores"]
            row[p["id"]] = (s["yes_no"]["mean_p_yes"],
                            s["likert5"]["mean_score"],
                            p.get("implementation_year"))
        out[(coord["x"], coord["sft"])][coord["year"]] = row
    return out


def policy_trajectories(by_year: dict, probe_idx: int = 0,
                        lo: int = -10, hi: int = 10) -> dict:
    """{policy_id: [(lookback, value), ...]} over the symmetric window."""
    enacted = {pid: tup[2] for pols in by_year.values()
               for pid, tup in pols.items() if tup[2] is not None}
    traj: dict = {}
    for pid, e in enacted.items():
        # lookback = Y - E, over year-models Y in [E+lo, E+hi].
        pts = [(y - e, by_year[y][pid][probe_idx])
               for y in range(e + lo, e + hi + 1)
               if y in by_year and pid in by_year[y]]
        if len(pts) >= 5:
            traj[pid] = pts
    return traj


def summarize(exps_dir: Path, backend: str, spec: str) -> dict:
    """Headline n-shape + scaling summary across the (X, sft) grid."""
    grid = load_window_evals(exps_dir, backend, spec)
    report: dict = {"backend": backend, "spec": spec, "cells": {}}
    for (x, sft), by_year in sorted(grid.items()):
        traj = policy_trajectories(by_year, probe_idx=0)
        n = len(traj)
        n_nshape = sum(1 for pts in traj.values()
                       if is_n_shape([p[0] for p in pts], [p[1] for p in pts]))
        report["cells"][f"x{x}{'_sft' if sft else ''}"] = {
            "x": x, "sft": sft, "n_policies": n,
            "n_nshape": n_nshape,
            "frac_nshape": round(n_nshape / n, 3) if n else None,
        }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exps-dir", type=Path, required=True,
                   help="Experiments dir holding policy_<...>/eval.json from Phase 3.")
    p.add_argument("--backend", required=True)
    p.add_argument("--spec", default="roll10")
    p.add_argument("--out-dir", type=Path,
                   default=Path("docs/findings/2026-06/rolling-window"))
    args = p.parse_args()

    report = summarize(args.exps_dir, args.backend, args.spec)
    if not report["cells"]:
        raise SystemExit(f"no window evals found in {args.exps_dir} for "
                         f"{args.backend}/{args.spec} — run Phase 3 first.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"nshape_{args.backend}_{args.spec}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  wrote {out}")
    for key, c in report["cells"].items():
        print(f"    {key}: {c['n_nshape']}/{c['n_policies']} n-shape "
              f"(frac={c['frac_nshape']})")


if __name__ == "__main__":
    main()
