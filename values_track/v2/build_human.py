"""Phase 1 — Build the clean HUMAN benchmark dataset from Kaushik's polling handoff.

Collapses the raw survey rows into ONE clean human answer distribution per
(question, year). No models involved; this is the reviewable ground truth that
Phases 2-3 consume. The whole "messiness" of the handoff is handled here, once:

  1. Keep geography = United States only        (drops the ~38% foreign rows).
  2. Keep the national TOPLINE demographic      (drops cross-tab subgroups).
  3. Exclude matrix/grid questions              (opaque column_N scale -> unscorable).
  4. Drop DK/refused/junk answers.
  5. Normalize option shares to sum to 1 / (question, year); drop multi-response.
  6. Keep questions with >= --min-years distinct survey years.

Outputs (to --out-dir, default D:/hist_LLM/policy_pred/values_v2/human):
  questions.csv        one row per question: item_id, source, source_question_id,
                       question_uid, question_text, answer_options, question_type,
                       n_years, y0, y1
  human_answers.csv    one row per (item_id, year, answer_option): human_share, n_size
                       (shares sum to 1 per item-year)  <- "how humans answered it"
  human_raw_all.parquet   ALL raw rows for the kept questions (incl foreign + subgroups)
  dropped_log.csv      question_uid + reason for every excluded question
  build_manifest.json  filter rule, kept/dropped counts, spot-checks

Usage:
  python values_track/v2/build_human.py --min-years 2
  python values_track/v2/build_human.py --min-years 1     # truly-all
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

HANDOFF = Path("C:/Users/danielyoon/Dropbox/SWM/kv_policy_events/policy_benchmark/"
               "handoffs/2026-07-03_policy_benchmark_handoff")
DEFAULT_OUT = Path("D:/hist_LLM/policy_pred/values_v2/human")

# Answer labels that are never substantive (excluded from options AND denominators).
# Reused from the pilot builder + a few extra DK/no-opinion variants seen across sources.
JUNK_EXACT = {"(dk)", "(refused)", "other", "none (volunteered)", "wouldnt vote",
              "dont know", "don't know", "refused", "no answer", "not applicable",
              "no opinion", "not sure", "undecided", "dk", "na", "no response",
              "don't know/refused", "no answer/don't know", "(vol.) dk", "vol. dk",
              "can't choose", "cant choose", "skipped", "no data", "unsure"}
JUNK_RE = re.compile(r"(don'?t know|no opinion|not sure|refused|no answer|undecided|"
                     r"can'?t choose|no response|not applicable|\bdk\b|\bvol\.)", re.I)

# Demographic labels that denote the national topline (not a cross-tab subgroup).
TOPLINE = {"aggregate", "respondents", "national", "u.s.", "us", "sample",
           "adults 18+", "adult (18+)", "total", "all", "all adults"}

MATRIX_RE = re.compile(r"(?i)^column_?\d+$")   # opaque matrix/grid columns


def is_junk(ans: str) -> bool:
    a = str(ans).strip().lower()
    return a in JUNK_EXACT or bool(JUNK_RE.search(a))


def slug(question_uid: str) -> str:
    """item_id derived from the (unique) question_uid -> guaranteed-unique join key."""
    return re.sub(r"[^0-9A-Za-z]+", "_", str(question_uid)).strip("_")


def load_raw(year_min: int, year_max: int) -> pd.DataFrame:
    cols = ["source", "source_question_id", "question_uid", "question_text", "period_year",
            "geography", "demographic", "demographic_value", "answer", "value", "value_unit", "n_size"]
    df = pd.read_csv(HANDOFF / "normalized" / "answer_observations.csv.gz", compression="gzip",
                     usecols=cols, dtype={"question_uid": str, "source_question_id": str}, low_memory=False)
    df["period_year"] = pd.to_numeric(df["period_year"], errors="coerce")
    df = df[df["period_year"].between(year_min, year_max)].copy()
    df["period_year"] = df["period_year"].astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-years", type=int, default=2, help="keep questions with >= this many survey years")
    ap.add_argument("--year-min", type=int, default=1943)
    ap.add_argument("--year-max", type=int, default=2020)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    dropped = []  # (question_uid, source, reason)

    raw = load_raw(args.year_min, args.year_max)
    n_raw_q = raw["question_uid"].nunique()

    # --- 1) geography = US -------------------------------------------------------------
    us = raw["geography"].astype(str).str.contains("United States", case=False, na=False)
    foreign_only = set(raw["question_uid"]) - set(raw.loc[us, "question_uid"])
    for q in foreign_only:
        dropped.append((q, "", "foreign_only"))
    df = raw[us].copy()

    # --- 2) national topline demographic ----------------------------------------------
    demo = df["demographic"].astype(str).str.strip().str.lower()
    df = df[demo.isin(TOPLINE)].copy()

    # --- 3) exclude matrix/grid questions (opaque column_N) ---------------------------
    is_matrix_row = df["demographic_value"].astype(str).str.match(MATRIX_RE)
    matrix_q = set(df.loc[is_matrix_row, "question_uid"])
    for q in matrix_q:
        dropped.append((q, "", "matrix_grid"))
    df = df[~df["question_uid"].isin(matrix_q)].copy()

    # --- 4) drop DK/junk answers ------------------------------------------------------
    df = df.dropna(subset=["value", "answer"])
    df = df[~df["answer"].map(is_junk)].copy()

    # --- collapse to one distribution per (question_uid, year) ------------------------
    # sum weight per answer within a (question, year); a stray duplicate row is summed.
    grp = (df.groupby(["question_uid", "period_year", "answer"], as_index=False)
             .agg(value=("value", "sum"), n_size=("n_size", "first")))

    q_meta = (df.sort_values("period_year")
                .groupby("question_uid")
                .agg(source=("source", "first"),
                     source_question_id=("source_question_id", "first"),
                     question_text=("question_text", "first"),
                     value_unit=("value_unit", "first")))          # indexed by question_uid

    # pre-filter to questions with enough survey years BEFORE the per-question loop (avoids O(N^2))
    yrs = grp.groupby("question_uid")["period_year"].nunique()
    src_map = q_meta["source"].to_dict()
    dropped.extend((q, src_map.get(q, ""), "below_min_years")
                   for q in yrs[yrs < args.min_years].index)
    grp = grp[grp["question_uid"].isin(set(yrs[yrs >= args.min_years].index))]

    questions, human_rows = [], []
    for quid, g in grp.groupby("question_uid", sort=False):
        meta = q_meta.loc[quid]
        item_id = slug(quid)
        # --- 5) multi-response guard: per-year substantive sum must look like a single choice
        yr_sums = g.groupby("period_year")["value"].sum()
        med = float(yr_sums.median())
        single = (50 <= med <= 150) or (0.5 <= med <= 1.5)   # percent-ish or fraction-ish
        if not single:
            dropped.append((quid, meta["source"], f"multi_response(sum~{med:.1f})"))
            continue
        # substantive option set (union across years), ordered by total weight desc
        opt_tot = g.groupby("answer")["value"].sum().sort_values(ascending=False)
        options = [str(a) for a in opt_tot.index]
        if len(options) < 2:
            dropped.append((quid, meta["source"], "degenerate(<2 options)"))
            continue
        years = sorted(g["period_year"].unique())
        # per-year normalized distribution
        for yr, gy in g.groupby("period_year"):
            tot = gy["value"].sum()
            if tot <= 0:
                continue
            nsz = pd.to_numeric(gy["n_size"].astype(str).str.replace(",", "", regex=False),
                                errors="coerce").dropna()
            nsz = int(nsz.iloc[0]) if len(nsz) else None
            for _, rr in gy.iterrows():
                human_rows.append(dict(item_id=item_id, question_uid=quid, year=int(yr),
                                       answer_option=str(rr["answer"]),
                                       human_share=round(float(rr["value"]) / tot, 5),
                                       n_size=nsz))
        questions.append(dict(
            item_id=item_id, source=meta["source"], source_question_id=meta["source_question_id"],
            question_uid=quid, question_text=str(meta["question_text"]),
            answer_options="|".join(options),
            question_type=("binary" if len(options) == 2 else "categorical"),
            n_years=len(years), y0=int(min(years)), y1=int(max(years)),
        ))

    q_df = pd.DataFrame(questions)
    h_df = pd.DataFrame(human_rows).sort_values(["item_id", "year", "answer_option"])
    assert q_df["item_id"].is_unique, "item_id collision (should be impossible from question_uid)"

    q_df.to_csv(out / "questions.csv", index=False)
    h_df.to_csv(out / "human_answers.csv", index=False)
    drop_df = pd.DataFrame(dropped, columns=["question_uid", "source", "reason"])
    drop_df.to_csv(out / "dropped_log.csv", index=False)
    # raw rows (incl foreign + subgroups) for the KEPT questions -> later cross-national checks
    kept = set(q_df["question_uid"])
    raw[raw["question_uid"].isin(kept)].to_parquet(out / "human_raw_all.parquet", index=False)

    manifest = dict(
        min_years=args.min_years, year_range=[args.year_min, args.year_max],
        filter="geography=US AND demographic in TOPLINE; drop matrix, DK/junk, multi-response, <2 options",
        raw_questions=int(n_raw_q), kept_questions=int(len(q_df)),
        human_answer_rows=int(len(h_df)),
        item_years=int(h_df[["item_id", "year"]].drop_duplicates().shape[0]),
        dropped_by_reason=drop_df["reason"].str.replace(r"\(.*\)", "", regex=True).value_counts().to_dict(),
        by_source=q_df["source"].value_counts().to_dict(),
        question_type=q_df["question_type"].value_counts().to_dict(),
    )
    (out / "build_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"kept {len(q_df):,} questions ({manifest['by_source']})")
    print(f"human_answers: {len(h_df):,} rows over {manifest['item_years']:,} (item,year) pairs -> {out}")
    print("dropped:", manifest["dropped_by_reason"])

    # spot-checks vs independently known values
    def share(qid_sub, year, opt_contains):
        m = h_df[(h_df["item_id"] == qid_sub) & (h_df["year"] == year)
                 & (h_df["answer_option"].str.contains(opt_contains, case=False, na=False))]
        return float(m["human_share"].sum()) if len(m) else float("nan")
    print("\nspot-checks:")
    print(f"  gss cappun(death pen) favor 1993 = {share('gss_cappun', 1993, 'favor'):.2f} (expect ~0.78)")
    print(f"  gss abany(abortion) yes 2018     = {share('gss_abany', 2018, 'yes'):.2f} (expect ~0.5)")


if __name__ == "__main__":
    main()
