"""Values-track report: model belief trends vs real survey trends.

Consumes the per-vintage values_battery eval.jsons (pulled from the pod), joins
them to the ground-truth table, and produces the track's deliverables:

  figures/values_grid.png        24-panel small-multiples: per item, the real
                                 survey focal share (blue, at actual survey waves)
                                 with the model's p_focal across all vintages
                                 (aqua) drawn over it — one line pair per arm run.
  figures/trend_corr.png         per-item Spearman trend correlation, with the
                                 permutation (drift) null band.
  values_model_tidy.csv          arm, vintage_year, item_id, p_focal
  values_summary.csv             per (arm, item): n_pairs, rho, null mean/sd, percentile
  README.md                      headline numbers + how to reproduce

Scoring pairs are EXACT-YEAR: each survey wave <= 2020 pairs with that same
year's vintage model (we have every year, so no wave-matching heuristics; waves
after 2020 are dropped — no model exists there). All vintages are still drawn on
the plots for a smooth model line; only exact pairs enter the statistics.

The drift null: per item, the same trend correlation computed against every
OTHER item's survey series (model series i x survey series j!=i). A matched
correlation only counts as signal if it beats what generic calendar drift
produces against arbitrary rising/falling survey lines.

Usage (after pulling evals locally):
    python analysis/values_report.py --evals caselaw=C:/tmp/values_evals/caselaw \\
        [--evals swmgst=C:/tmp/values_evals/swmgst] [--out docs/findings/2026-07/values_pilot]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
BATTERY = REPO_ROOT / "us_values_battery_v1.csv"
TRUTH = REPO_ROOT / "data_artifacts" / "values_truth_v1.csv"
MODEL_END_YEAR = 2020  # vintages stop here; later survey waves have no partner

# dataviz reference palette (light mode): series + chrome.
BLUE, AQUA, RED = "#2a78d6", "#1baf7a", "#e34948"
SURFACE, INK, SEC, MUT, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
ARM_COLORS = [AQUA, "#eda100"]  # model line per arm (slot 2, then slot 3)


def load_battery() -> dict[str, dict]:
    with open(BATTERY, encoding="utf-8") as f:
        return {r["item_id"]: r for r in csv.DictReader(f)}


def load_truth() -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {}
    with open(TRUTH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            series.setdefault(r["item_id"], []).append((int(r["year"]), float(r["focal_share"])))
    return {k: sorted(v) for k, v in series.items()}


def load_evals(root: Path) -> dict[int, dict[str, float]]:
    """{vintage_year: {item_id: p_focal}} from eval.jsons under root."""
    out: dict[int, dict[str, float]] = {}
    for f in glob.glob(str(root / "**" / "eval.json"), recursive=True):
        name = os.path.basename(os.path.dirname(f))
        m = re.search(r"_(\d{4})_", name) or re.search(r"(\d{4})", name)
        if not m:
            continue
        d = json.load(open(f, encoding="utf-8"))
        items = (d.get("results") or {}).get("items") or d.get("items") or []
        out[int(m.group(1))] = {it["item_id"]: float(it["p_focal"]) for it in items}
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rho without scipy: Pearson on average ranks."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def pairs_for(model: dict[int, dict[str, float]], truth_series: list[tuple[int, float]],
              item_id: str) -> tuple[list[float], list[float]]:
    """Exact-year (model_p, survey_share) pairs at survey waves <= MODEL_END_YEAR."""
    xs, ys = [], []
    for year, share in truth_series:
        if year <= MODEL_END_YEAR and year in model and item_id in model[year]:
            xs.append(model[year][item_id])
            ys.append(share)
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evals", action="append", required=True, metavar="ARM=DIR",
                    help="arm label + local dir containing that arm's eval.jsons (repeatable)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs/findings/2026-07/values_pilot")
    args = ap.parse_args()

    battery = load_battery()
    truth = load_truth()
    arms: dict[str, dict[int, dict[str, float]]] = {}
    for spec in args.evals:
        arm, _, d = spec.partition("=")
        arms[arm] = load_evals(Path(d))
        print(f"arm {arm}: {len(arms[arm])} vintages loaded")

    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    items = list(battery)

    # ---- tidy CSV ----
    with open(args.out / "values_model_tidy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "vintage_year", "item_id", "p_focal"])
        for arm, model in arms.items():
            for y in sorted(model):
                for iid, p in model[y].items():
                    w.writerow([arm, y, iid, round(p, 4)])

    # ---- statistics: matched rho + drift null (per arm, per item) ----
    summary_rows = []
    for arm, model in arms.items():
        for iid in items:
            xs, ys = pairs_for(model, truth[iid], iid)
            rho = spearman(xs, ys) if len(xs) >= 5 else float("nan")
            nulls = []
            for jid in items:                       # model series i vs survey series j != i
                if jid == iid:
                    continue
                nx, ny = [], []
                for year, share in truth[jid]:
                    if year <= MODEL_END_YEAR and year in model and iid in model[year]:
                        nx.append(model[year][iid])
                        ny.append(share)
                if len(nx) >= 5:
                    nulls.append(spearman(nx, ny))
            nmean = st.mean(nulls) if nulls else float("nan")
            nsd = st.pstdev(nulls) if len(nulls) > 1 else float("nan")
            pct = (sum(1 for n in nulls if n < rho) / len(nulls) * 100
                   if nulls and rho == rho else float("nan"))
            summary_rows.append(dict(arm=arm, item_id=iid, n_pairs=len(xs),
                                     rho=round(rho, 3) if rho == rho else "",
                                     null_mean=round(nmean, 3) if nmean == nmean else "",
                                     null_sd=round(nsd, 3) if nsd == nsd else "",
                                     pct_beats_null=round(pct, 1) if pct == pct else ""))
    with open(args.out / "values_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)

    # ---- FIG 1: 24-panel two-line grid ----
    plt.rcParams.update({"font.family": "sans-serif", "text.color": INK,
                         "axes.edgecolor": BASE, "axes.labelcolor": SEC,
                         "xtick.color": MUT, "ytick.color": MUT})
    fig, axes = plt.subplots(4, 6, figsize=(20, 11), sharex=True, sharey=True,
                             facecolor=SURFACE)
    for ax, iid in zip(axes.flat, items):
        ax.set_facecolor(SURFACE)
        sy = [y for y, _ in truth[iid]]
        sv = [v * 100 for _, v in truth[iid]]
        ax.plot(sy, sv, "-o", ms=2.5, lw=2, color=BLUE, zorder=3)
        for k, (arm, model) in enumerate(arms.items()):
            my = sorted(y for y in model if iid in model[y])
            mv = [model[y][iid] * 100 for y in my]
            ax.plot(my, mv, "-", lw=2, color=ARM_COLORS[k % len(ARM_COLORS)], zorder=2)
        ev = str(battery[iid]["paired_event_ids"])
        m = re.search(r"US-(\d{4})-", ev)
        if m:
            ax.axvline(int(m.group(1)), color=MUT, lw=1, ls=(0, (4, 3)), zorder=1)
        row = next(r for r in summary_rows if r["item_id"] == iid)
        ax.set_title(f"{iid}  (rho={row['rho']}, n={row['n_pairs']})", fontsize=8.5,
                     loc="left", color=INK)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        ax.tick_params(labelsize=7, length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = ([plt.Line2D([], [], color=BLUE, marker="o", ms=3, lw=2, label="real survey (GSS/Gallup)")]
               + [plt.Line2D([], [], color=ARM_COLORS[k % len(ARM_COLORS)], lw=2,
                             label=f"model p_focal ({arm})") for k, arm in enumerate(arms)])
    fig.legend(handles=handles, loc="upper right", fontsize=9, frameon=False,
               bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("Values track: model belief across vintages vs real survey trend "
                 "(% giving the focal answer; dashed line = paired policy event)",
                 fontsize=13, x=0.02, ha="left", fontweight="bold", color=INK)
    fig.supxlabel("year (survey wave / model vintage)", color=SEC)
    fig.supylabel("% focal answer", color=SEC)
    fig.tight_layout(rect=[0.01, 0.01, 1, 0.96])
    fig.savefig(args.out / "figures" / "values_grid.png", dpi=140, facecolor=SURFACE)
    plt.close(fig)

    # ---- FIG 2: per-item trend correlation vs drift null ----
    for arm in arms:
        rows = [r for r in summary_rows if r["arm"] == arm and r["rho"] != ""]
        rows.sort(key=lambda r: float(r["rho"]), reverse=True)
        fig, ax = plt.subplots(figsize=(10, 7.5), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        ys = range(len(rows))
        vals = [float(r["rho"]) for r in rows]
        ax.barh(list(ys), vals, height=0.62,
                color=[BLUE if v >= 0 else RED for v in vals], zorder=3)
        for i, r in enumerate(rows):
            if r["null_mean"] != "" and r["null_sd"] != "":
                nm, ns = float(r["null_mean"]), float(r["null_sd"])
                ax.plot([nm - ns, nm + ns], [i, i], color=MUT, lw=3, alpha=.55, zorder=2)
                ax.plot([nm], [i], "o", color=SEC, ms=4, zorder=4)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([r["item_id"] for r in rows], fontsize=8, color=SEC)
        ax.invert_yaxis()
        ax.axvline(0, color=BASE, lw=1)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Spearman rho: model p_focal vs real survey share (exact-year pairs)",
                      color=SEC, fontsize=9)
        ax.set_title(f"Trend correlation per item — arm: {arm}\n"
                     f"gray band = drift null (same model series vs the other items' survey lines)",
                     loc="left", fontsize=11, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
        ax.tick_params(length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(args.out / "figures" / f"trend_corr_{arm}.png", dpi=140, facecolor=SURFACE)
        plt.close(fig)

    # ---- README ----
    lines = ["# Values pilot — model belief trends vs real survey trends\n"]
    for arm in arms:
        rr = [float(r["rho"]) for r in summary_rows if r["arm"] == arm and r["rho"] != ""]
        bb = [float(r["pct_beats_null"]) for r in summary_rows
              if r["arm"] == arm and r["pct_beats_null"] != ""]
        if rr:
            lines.append(f"- **{arm}**: median rho = {st.median(rr):+.2f} over {len(rr)} items; "
                         f"median null-percentile = {st.median(bb):.0f}%\n")
    lines += ["\nFigures: `figures/values_grid.png` (24-panel two-line grid), "
              "`figures/trend_corr_<arm>.png`.\n",
              "Data: `values_model_tidy.csv`, `values_summary.csv` "
              "(truth: `data_artifacts/values_truth_v1.csv`).\n",
              "\nReproduce: `python analysis/values_report.py --evals <arm>=<dir>`\n"]
    (args.out / "README.md").write_text("".join(lines), encoding="utf-8")
    print(f"wrote report -> {args.out}")


if __name__ == "__main__":
    main()
