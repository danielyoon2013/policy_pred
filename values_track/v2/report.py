"""Phase 3 — Compare model answer distributions to humans (point-in-time).

Joins each vintage model's `p_per_option` (from the Phase-2 eval.jsons) to the
Phase-1 human distribution on (item_id, year == model_year), and scores:
  - argmax agreement : model's top-scored option == human plurality answer (headline;
                       robust to LLM miscalibration).
  - TV distance      : total-variation distance between the model and human
                       distributions over the shared option set (secondary).

Outputs (to --out, default D:/hist_LLM/policy_pred/values_v2/report):
  per_item.csv          one row per (arm, item, year): argmax_match, tv_distance, ...
  agreement_by_year.csv per (arm, year): n, argmax_agreement, mean_tv, chance
  figures/agreement_by_year.png

Usage:
  python values_track/v2/report.py \
      --human-dir D:/hist_LLM/policy_pred/values_v2/human \
      --evals caselaw=D:/hist_LLM/policy_pred/values_v2/eval/caselaw \
      --evals swmgst=D:/hist_LLM/policy_pred/values_v2/eval/swmgst
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd

DEF_HUMAN = Path("D:/hist_LLM/policy_pred/values_v2/human")
DEF_OUT = Path("D:/hist_LLM/policy_pred/values_v2/report")
_YEAR = re.compile(r"_(\d{4})_")


def load_human(human_dir: Path) -> dict:
    h = pd.read_csv(human_dir / "human_answers.csv")
    human = {}
    for (item, yr), g in h.groupby(["item_id", "year"]):
        human[(item, int(yr))] = dict(zip(g["answer_option"].astype(str), g["human_share"].astype(float)))
    return human


def load_eval_arm(eval_dir: str) -> dict:
    """(item_id, model_year) -> {option: model_prob} from every eval.json under eval_dir."""
    out = {}
    for f in glob.glob(str(Path(eval_dir) / "**" / "eval.json"), recursive=True):
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        name = data.get("experiment", "") or Path(f).parent.name
        m = _YEAR.search(name) or re.search(r"(\d{4})", name)
        if not m:
            continue
        yr = int(m.group(1))
        items = (data.get("results") or {}).get("items") or data.get("items") or []
        for it in items:
            ppo = it.get("p_per_option") or {}
            if ppo:
                out[(it["item_id"], yr)] = {str(k): float(v) for k, v in ppo.items()}
    return out


def tv_distance(p: dict, q: dict, keys) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human-dir", type=Path, default=DEF_HUMAN)
    ap.add_argument("--evals", action="append", required=True, metavar="ARM=DIR",
                    help="arm label = eval dir; repeatable")
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    args = ap.parse_args()

    human = load_human(args.human_dir)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in args.evals:
        arm, d = spec.split("=", 1)
        model = load_eval_arm(d)
        for (item, yr), mp in model.items():
            hp = human.get((item, yr))
            if not hp or not mp:
                continue
            keys = set(hp) | set(mp)
            rows.append(dict(
                arm=arm, item_id=item, year=yr,
                model_argmax=max(mp, key=mp.get), human_plurality=max(hp, key=hp.get),
                argmax_match=int(max(mp, key=mp.get) == max(hp, key=hp.get)),
                tv_distance=round(tv_distance(mp, hp, keys), 4),
                n_options=len(keys), chance=round(1.0 / len(keys), 4),
            ))
    if not rows:
        print("no (item,year) pairs matched between evals and human data — check paths/naming.")
        return
    per = pd.DataFrame(rows)
    per.to_csv(args.out / "per_item.csv", index=False)

    byyear = (per.groupby(["arm", "year"])
                 .agg(n=("argmax_match", "size"), argmax_agreement=("argmax_match", "mean"),
                      mean_tv=("tv_distance", "mean"), chance=("chance", "mean"))
                 .reset_index())
    byyear.to_csv(args.out / "agreement_by_year.csv", index=False)

    print("overall (model top == human plurality):")
    print(per.groupby("arm").agg(n=("argmax_match", "size"),
                                 argmax_agreement=("argmax_match", "mean"),
                                 mean_tv=("tv_distance", "mean"),
                                 chance=("chance", "mean")).round(3).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5))
    for arm, g in byyear.sort_values("year").groupby("arm"):
        ax.plot(g["year"], g["argmax_agreement"], "-o", ms=3, label=arm)
    ch = byyear.groupby("year")["chance"].mean()
    ax.plot(ch.index, ch.values, "--", color="gray", lw=1, label="chance (mean 1/n_options)")
    ax.set_xlabel("vintage model year"); ax.set_ylabel("argmax agreement (model top == human plurality)")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Model vs human: point-in-time answer agreement by vintage year")
    fig.tight_layout()
    fig.savefig(args.out / "figures" / "agreement_by_year.png", dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
