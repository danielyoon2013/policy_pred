"""Build the values-track battery + ground-truth table from Kaushik's polling handoff.

The values track probes year-vintage models with real survey questions (as a period
respondent) and grades their answer-probability trends against real GSS/Gallup
marginals. This script turns the curated 24-item pilot shortlist into the two files
the track runs on:

  us_values_battery_v1.csv          (repo root)   one row per survey item: the
      respondent-facing wording, the survey's own answer options, and which
      option(s) count as the focal answer.
  data_artifacts/values_truth_v1.csv               the answer key: observed focal
      share per (item, survey year), from the handoff's weighted answer rows.

Everything judgment-laden lives in ITEM_SPECS below (wording, option mapping,
focal/direction rule per item) so the hand-review gate is one dict in one file.
Conventions: shares are computed over substantive answers only (DK/refused/other
excluded); GSS scale items (helpblk/helpsick/eqwlth) present only the labeled
anchor positions to the model while the truth share assigns the unlabeled numeric
scale points to their side of the midpoint (see each item's direction_note).

Usage:
    python scripts/data_prep/build_values_battery.py            # build + validate
    python scripts/data_prep/build_values_battery.py --inspect  # just print observed labels
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HANDOFF = Path("C:/Users/danielyoon/Dropbox/SWM/kv_policy_events/policy_benchmark/"
               "handoffs/2026-07-03_policy_benchmark_handoff")
PILOT = Path("C:/Users/danielyoon/Dropbox/SWM/values_track/pilot/values_pilot_candidates.csv")
OUT_BATTERY = REPO_ROOT / "us_values_battery_v1.csv"
OUT_TRUTH = REPO_ROOT / "data_artifacts" / "values_truth_v1.csv"

# Answer labels that are never substantive (excluded from options AND denominators).
JUNK = {"(dk)", "(refused)", "other", "none (volunteered)", "wouldnt vote",
        "dont know", "don't know", "refused", "no answer", "not applicable"}

# Per-item spec. `options` is an ordered list of (shown_label, [data_labels]) —
# shown_label is what the model sees as an answer choice; data_labels are the raw
# handoff answer strings it aggregates. `focal` lists the shown_labels whose
# probability/share is the tracked quantity. `truth_focal_extra` /
# `truth_denom_extra` fold unlabeled scale points into the truth share only.
ITEM_SPECS = {
    ("gss", "racmar"): dict(
        wording="Do you think there should be laws against marriages between Blacks and whites?",
        options=[("Yes", ["yes"]), ("No", ["no"])], focal=["Yes"],
        note="Yes = favors a legal ban on interracial marriage (declines 37%->10%, 1972-2002)."),
    ("gss", "homosex"): dict(
        wording=("What about sexual relations between two adults of the same sex — do you think it is "
                 "always wrong, almost always wrong, wrong only sometimes, or not wrong at all?"),
        options=[("Always wrong", ["always wrong"]), ("Almost always wrong", ["almost always wrong"]),
                 ("Wrong only sometimes", ["wrong only sometimes"]), ("Not wrong at all", ["not wrong at all"])],
        focal=["Always wrong"],
        note="'Always wrong' tracks traditional morality; falls 73%->34% (1973-2024)."),
    ("gss", "colhomo"): dict(
        wording=("Should a man who admits that he is a homosexual be allowed to teach in a college "
                 "or university, or not?"),
        options=[("Yes, allowed to teach", ["yes, allowed to teach"]), ("Not allowed", ["not allowed"])],
        focal=["Yes, allowed to teach"],
        note="Civil-liberties framing of gay rights."),
    ("gss", "abany"): dict(
        wording=("Please tell me whether or not you think it should be possible for a pregnant woman "
                 "to obtain a legal abortion if the woman wants it for any reason?"),
        options=[("Yes", ["yes"]), ("No", ["no"])], focal=["Yes"],
        note="The maximal abortion-rights item (37%->58%, 1977-2024)."),
    ("gss", "abhlth"): dict(
        wording=("Please tell me whether or not you think it should be possible for a pregnant woman "
                 "to obtain a legal abortion if the woman's own health is seriously endangered by "
                 "the pregnancy?"),
        options=[("Yes", ["yes"]), ("No", ["no"])], focal=["Yes"],
        note="High-consensus abortion item (~90% yes throughout) — a level anchor."),
    ("gallup", "971"): dict(
        wording=("Do you think abortions should be legal under any circumstances, legal only under "
                 "certain circumstances, or illegal in all circumstances?"),
        options=[("Legal under any circumstances", ["Legal under any"]),
                 ("Legal only under certain circumstances", ["Legal under certain"]),
                 ("Illegal in all circumstances", ["Illegal in all"])],
        focal=["Legal under any circumstances"],
        note="Modern-era abortion legality (2001-2025)."),
    ("gss", "fepol"): dict(
        wording=("Tell me if you agree or disagree with this statement: Most men are better suited "
                 "emotionally for politics than are most women."),
        options=[("Agree", ["agree"]), ("Disagree", ["disagree"])], focal=["Disagree"],
        note="Disagree = endorses gender equality in politics (53%->82%, 1974-2024)."),
    ("gss", "fepres"): dict(
        wording=("If your party nominated a woman for President, would you vote for her if she were "
                 "qualified for the job?"),
        options=[("Yes", ["yes"]), ("No", ["no"])], focal=["Yes"],
        note="'wouldnt vote' excluded as non-substantive. 1972-2010 only."),
    ("gss", "cappun"): dict(
        wording="Do you favor or oppose the death penalty for persons convicted of murder?",
        options=[("Favor", ["favor"]), ("Oppose", ["oppose"])], focal=["Favor"],
        note="NON-MONOTONE: support peaks ~1990s then falls — tests reversal tracking."),
    ("gallup", "1020"): dict(
        wording="Are you in favor of the death penalty for a person convicted of murder?",
        options=[("Yes, in favor", ["Yes, in favor"]), ("No, not in favor", ["No, not in favor"])],
        focal=["Yes, in favor"],
        note="Gallup twin of cappun (2000-2025)."),
    ("gss", "gunlaw"): dict(
        wording=("Would you favor or oppose a law which would require a person to obtain a police "
                 "permit before he or she could buy a gun?"),
        options=[("Favor", ["favor"]), ("Oppose", ["oppose"])], focal=["Favor"],
        note="Gun permits; stable-high support (~70%)."),
    ("gallup", "1042"): dict(
        wording=("In general, do you feel that the laws covering the sale of firearms should be made "
                 "more strict, less strict, or kept as they are now?"),
        options=[("More strict", ["More strict"]), ("Less strict", ["Less strict"]),
                 ("Kept as they are now", ["Kept as they are now"])],
        focal=["More strict"],
        note="Modern gun-law item (2001-2025)."),
    ("gss", "prayer"): dict(
        wording=("The United States Supreme Court has ruled that no state or local government may "
                 "require the reading of the Lord's Prayer or Bible verses in public schools. Do you "
                 "approve or disapprove of the court ruling?"),
        options=[("Approve", ["approve"]), ("Disapprove", ["disapprove"])], focal=["Disapprove"],
        note="Disapprove = endorses school prayer (traditional-religion side); pairs with Engel v. Vitale."),
    ("gss", "natrace"): dict(
        wording=("We are faced with many problems in this country, none of which can be solved easily "
                 "or inexpensively. Are we spending too much money, too little money, or about the "
                 "right amount on improving the conditions of Black Americans?"),
        options=[("Too little", ["too little"]), ("About right", ["about right"]), ("Too much", ["too much"])],
        focal=["Too little"],
        note="National-spending frame on race."),
    ("gss", "helpblk"): dict(
        wording=("Some people think that Black Americans have been discriminated against for so long "
                 "that the government has a special obligation to help improve their living standards; "
                 "others believe that the government should not be giving special treatment to Black "
                 "Americans. Where would you place yourself?"),
        options=[("The government should help", ["government should help"]),
                 ("I agree with both positions", ["agree with both"]),
                 ("No special treatment", ["no special treatment"])],
        focal=["The government should help"],
        truth_focal_extra=["2.0"], truth_denom_extra=["2.0", "4.0"],
        note=("5-pt scale; model sees the 3 labeled anchors. Truth: focal side = {1,2}, denominator = "
              "all 5 points (unlabeled 2/4 folded to their side of the midpoint).")),
    ("gss", "busing"): dict(
        wording=("In general, do you favor or oppose the busing of Black and white school children "
                 "from one school district to another?"),
        options=[("Favor", ["favor"]), ("Oppose", ["oppose"])], focal=["Favor"],
        note="Low-support item (unpopular-policy tracking), 1972-1996 only."),
    ("gss", "natfare"): dict(
        wording=("We are faced with many problems in this country, none of which can be solved easily "
                 "or inexpensively. Are we spending too much money, too little money, or about the "
                 "right amount on welfare?"),
        options=[("Too little", ["too little"]), ("About right", ["about right"]), ("Too much", ["too much"])],
        focal=["Too much"],
        note="NON-MONOTONE: 'too much' peaks mid-1990s at PRWORA."),
    ("gss", "helpsick"): dict(
        wording=("Some people think that it is the responsibility of the government in Washington to "
                 "help people pay for doctor and hospital bills; others think that these matters are "
                 "not the responsibility of the federal government and that people should take care of "
                 "these things themselves. Where would you place yourself?"),
        options=[("The government should help", ["government should help"]),
                 ("I agree with both positions", ["agree with both"]),
                 ("People should take care of these things themselves", ["people should care for themselves"])],
        focal=["The government should help"],
        truth_focal_extra=["2.0"], truth_denom_extra=["2.0", "4.0"],
        note="5-pt scale, same anchor treatment as helpblk. Pairs with Medicare/ACA."),
    ("gss", "natenvir"): dict(
        wording=("We are faced with many problems in this country, none of which can be solved easily "
                 "or inexpensively. Are we spending too much money, too little money, or about the "
                 "right amount on improving and protecting the environment?"),
        options=[("Too little", ["too little"]), ("About right", ["about right"]), ("Too much", ["too much"])],
        focal=["Too little"],
        note="Environmental spending."),
    ("gss", "letin1a"): dict(
        wording=("Do you think the number of immigrants to America nowadays should be increased a lot, "
                 "increased a little, remain the same as it is, reduced a little, or reduced a lot?"),
        options=[("Increased a lot", ["increased a lot"]), ("Increased a little", ["increased a little"]),
                 ("Remain the same as it is", ["remain the same as it is"]),
                 ("Reduced a little", ["reduced a little"]), ("Reduced a lot", ["reduced a lot"])],
        focal=["Increased a lot", "Increased a little"],
        note="Focal = the two 'increased' options summed. Short window (2004-2024)."),
    ("gallup", "1062"): dict(
        wording="Do you think the use of marijuana should be made legal, or not?",
        options=[("Yes, legal", ["Yes, legal"]), ("No, not legal", ["No, not legal"])],
        focal=["Yes, legal"],
        note="Marijuana legalization (2000-2025); pairs with the Controlled Substances Act."),
    ("gss", "conlegis"): dict(
        wording=("As far as the people running Congress are concerned, would you say you have a great "
                 "deal of confidence, only some confidence, or hardly any confidence at all in them?"),
        options=[("A great deal", ["a great deal"]), ("Only some", ["only some"]),
                 ("Hardly any", ["hardly any"])],
        focal=["A great deal"],
        note="Cross-cutting trust in government."),
    ("gallup", "1092"): dict(
        wording=("Please tell me how much confidence you, yourself, have in the U.S. Supreme Court — "
                 "a great deal, quite a lot, some, or very little?"),
        options=[("A great deal", ["A great deal"]), ("Quite a lot", ["Quite a lot"]),
                 ("Some", ["Some"]), ("Very little", ["Very little"])],
        focal=["A great deal", "Quite a lot"],
        note="Longest series in the pilot (1973-2025). Focal = top-two confidence, Gallup's own headline metric."),
    ("gss", "eqwlth"): dict(
        wording=("Some people think that the government in Washington ought to reduce the income "
                 "differences between the rich and the poor, perhaps by raising the taxes of wealthy "
                 "families or by giving income assistance to the poor. Others think that the government "
                 "should not concern itself with reducing this income difference between the rich and "
                 "the poor. Which position comes closer to the way you feel?"),
        options=[("The government should reduce income differences",
                  ["the government should reduce income differences"]),
                 ("The government should not concern itself with reducing income differences",
                  ["the government should not concern itself with reducing income differences"])],
        focal=["The government should reduce income differences"],
        truth_focal_extra=["2.0", "3.0"], truth_denom_extra=["2.0", "3.0", "4.0", "5.0", "6.0"],
        note=("7-pt scale; model sees the two labeled ends. Truth: focal side = {1,2,3}, denominator = "
              "all 7 points.")),
}


def load_answers():
    import pandas as pd
    ans = pd.read_csv(HANDOFF / "normalized" / "answer_observations.csv.gz", compression="gzip",
                      usecols=["source", "question_uid", "answer", "value", "period_year"],
                      dtype={"question_uid": str}, low_memory=False)
    ans = ans[ans["source"].isin(["gss", "gallup"])].copy()
    # `value` is stored as string in the handoff (mixed-type column); coerce.
    ans["value"] = pd.to_numeric(ans["value"], errors="coerce")
    return ans.dropna(subset=["value"])


def main() -> None:
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inspect", action="store_true",
                    help="print observed answer labels per pilot item and exit")
    args = ap.parse_args()

    pilot = pd.read_csv(PILOT)
    pilot["question_uid"] = pilot["question_uid"].astype(str)
    ans = load_answers()

    if args.inspect:
        for _, r in pilot.iterrows():
            sub = ans[(ans["source"] == r["source"]) & (ans["question_uid"] == r["question_uid"])]
            labels = sub.groupby("answer")["value"].sum().sort_values(ascending=False)
            print(f"{r['source']}/{r['source_question_id']}: {list(labels.index)}")
        return

    battery_rows, truth_rows, problems = [], [], []
    for _, r in pilot.iterrows():
        key = (r["source"], str(r["source_question_id"]).lower())
        spec = ITEM_SPECS.get(key)
        if spec is None:
            problems.append(f"NO SPEC for {key}")
            continue
        item_id = f"{key[0]}_{key[1]}"
        sub = ans[(ans["source"] == r["source"]) & (ans["question_uid"] == r["question_uid"])]
        observed = set(sub["answer"].unique())

        # Validate that every mapped data label actually occurs.
        mapped = [dl for _, dls in spec["options"] for dl in dls]
        mapped += spec.get("truth_focal_extra", []) + spec.get("truth_denom_extra", [])
        missing = [m for m in mapped if m not in observed]
        if missing:
            problems.append(f"{item_id}: mapped labels not in data: {missing}")
        unexplained = [o for o in observed
                       if o not in mapped and o.lower() not in JUNK]
        if unexplained:
            problems.append(f"{item_id}: observed labels neither mapped nor junk: {unexplained}")

        focal_data = {dl for shown, dls in spec["options"] if shown in spec["focal"] for dl in dls}
        focal_data |= set(spec.get("truth_focal_extra", []))
        denom_data = {dl for _, dls in spec["options"] for dl in dls}
        denom_data |= set(spec.get("truth_denom_extra", []))

        battery_rows.append(dict(
            item_id=item_id, source=r["source"], source_question_id=r["source_question_id"],
            question_uid=r["question_uid"], question_text=spec["wording"],
            answer_options="|".join(shown for shown, _ in spec["options"]),
            focal_answer=" + ".join(spec["focal"]),
            direction_note=spec["note"],
            value_construct=r["value_construct"], paired_event_ids=r["paired_event_ids"],
            y0=int(r["y0"]), y1=int(r["y1"]), n_years=int(r["n_years"]),
        ))

        per_year = sub[sub["answer"].isin(denom_data)].groupby("period_year")
        for year, g in per_year:
            tot = g["value"].sum()
            foc = g[g["answer"].isin(focal_data)]["value"].sum()
            if tot > 0:
                truth_rows.append(dict(item_id=item_id, question_uid=r["question_uid"],
                                       year=int(year), focal_share=round(foc / tot, 4),
                                       total_weight=round(float(tot), 2)))

    battery = pd.DataFrame(battery_rows)
    truth = pd.DataFrame(truth_rows).sort_values(["item_id", "year"])
    OUT_TRUTH.parent.mkdir(parents=True, exist_ok=True)
    battery.to_csv(OUT_BATTERY, index=False)
    truth.to_csv(OUT_TRUTH, index=False)

    print(f"battery: {len(battery)} items -> {OUT_BATTERY}")
    print(f"truth:   {len(truth)} (item,year) rows -> {OUT_TRUTH}")
    if problems:
        print("\nPROBLEMS (fix ITEM_SPECS before review):")
        for p in problems:
            print("  -", p)
        sys.exit(1)

    # Spot-checks against independently verified numbers.
    def share(item, year):
        m = truth[(truth["item_id"] == item) & (truth["year"] == year)]
        return float(m["focal_share"].iloc[0]) if len(m) else float("nan")
    print("\nspot-checks (expected from prior verification):")
    print(f"  racmar 1972 = {share('gss_racmar', 1972):.2f} (expect ~0.37)   "
          f"2002 = {share('gss_racmar', 2002):.2f} (expect ~0.10)")
    print(f"  fepol  1974 = {share('gss_fepol', 1974):.2f} (expect ~0.53 disagree) "
          f"2024 = {share('gss_fepol', 2024):.2f} (expect ~0.82)")
    print(f"  cappun 1974 = {share('gss_cappun', 1974):.2f}  1994 = {share('gss_cappun', 1994):.2f} "
          f"(hump: 1994 should exceed 1974)  2024 = {share('gss_cappun', 2024):.2f}")


if __name__ == "__main__":
    main()
