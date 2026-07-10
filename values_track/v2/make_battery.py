"""Phase 2a — Derive the model-run battery from the Phase-1 human dataset.

Reads questions.csv + human_answers.csv (Phase 1) and writes battery.csv in the
schema `evaluators/value/survey.py` expects (item_id, question_uid, question_text,
answer_options, focal_answer, value_construct, paired_event_ids). `focal_answer` is
set to the OVERALL human-plurality option purely to satisfy survey.py's schema
(focal must be a valid option); the actual model<->human comparison in Phase 3 uses
the full `p_per_option`, not the focal. value_construct/paired_event_ids are left
empty (unused in this simple point-in-time comparison).

Usage:
  python values_track/v2/make_battery.py \
      --human-dir D:/hist_LLM/policy_pred/values_v2/human \
      --out       D:/hist_LLM/policy_pred/values_v2/battery/battery.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEF_HUMAN = Path("D:/hist_LLM/policy_pred/values_v2/human")
DEF_OUT = Path("D:/hist_LLM/policy_pred/values_v2/battery/battery.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human-dir", type=Path, default=DEF_HUMAN)
    ap.add_argument("--out", type=Path, default=DEF_OUT)
    args = ap.parse_args()

    q = pd.read_csv(args.human_dir / "questions.csv")
    h = pd.read_csv(args.human_dir / "human_answers.csv")

    # overall human-plurality option per item (max total share across years) -> focal_answer
    plurality = (h.groupby(["item_id", "answer_option"])["human_share"].sum()
                   .reset_index()
                   .sort_values("human_share", ascending=False)
                   .drop_duplicates("item_id")
                   .set_index("item_id")["answer_option"].to_dict())

    rows = []
    for _, r in q.iterrows():
        opts = str(r["answer_options"]).split("|")
        focal = plurality.get(r["item_id"], opts[0])
        if focal not in opts:                      # safety: focal must be a valid option
            focal = opts[0]
        rows.append(dict(
            item_id=r["item_id"], source=r["source"], source_question_id=r["source_question_id"],
            question_uid=r["question_uid"], question_text=r["question_text"],
            answer_options=r["answer_options"], focal_answer=focal,
            value_construct="", paired_event_ids="",
            question_type=r["question_type"], n_years=r["n_years"], y0=r["y0"], y1=r["y1"],
        ))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    bat = pd.DataFrame(rows)
    bat.to_csv(args.out, index=False)
    # also drop a repo copy so `git pull` brings the battery onto the pod
    repo_copy = Path(__file__).resolve().parent / "battery.csv"
    bat.to_csv(repo_copy, index=False)
    print(f"battery: {len(bat)} items -> {args.out}")
    print(f"         repo copy -> {repo_copy}  (git-add for the pod sweep)")
    print("  (focal_answer = human plurality; comparison uses full p_per_option in Phase 3)")


if __name__ == "__main__":
    main()
