"""synth/sft_corpus: per-year corpus-grounded SFT pairs for the policy probe.

Round-2 replacement for synth/sft_format. Instead of generic hand-templated
topics forced to a uniform label distribution, this reads each year's own legal
corpus and asks the model to produce a normative "Should ...?" question plus the
answer the passage's stance supports — on both a Yes/No and a 5-point Likert
scale. The label distribution therefore EMERGES from the corpus (no
trim_to_counts), which is the whole point: Round 1's forced 20%x5 Likert
distribution is the suspected cause of the Likert-probe collapse.

One passage -> one question -> two probe-shaped chat records (one Yes/No, one
Likert), rendered byte-identically to evaluators/policy_battery so SFT-on-top
teaches the exact eval format. Output is per year (sft_corpus/<Y>.jsonl), a
fixed small set used as the SFT-on-top stage for every year-Y window model.

Seeds are the first `n_seeds` legal passages in windowing.stable_order — the
same canonical order the LoRA pool draws from — so the SFT passages are a strict
prefix of the LoRA seeds (SFT seeds subset of LoRA seeds). Keep `seed` in sync
with synth/naive.

Usage:
    export OPENAI_API_KEY=sk-...
    python synth/sft_corpus/run.py --year 1935
    python synth/sft_corpus/run.py --year 1935 --n-seeds 50 --out /tmp/smoke.jsonl   # smoke
    python synth/sft_corpus/run.py --years 1931-2019                                  # full
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Make the repo root importable (for windowing) regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import windowing  # noqa: E402  (repo-root module)
# Single source of truth for probe prompt formatting — the SFT records and the
# eval prompts MUST be byte-identical up to the answer position, so the SFT
# formatter reuses the exact builders the eval scores against. Direct records
# use the bare builders; CoT records use the reason-then-answer builders.
from evaluators.policy_battery import (  # noqa: E402
    _build_yesno_prompt, _build_likert_prompt,
    build_yesno_cot_user, build_likert_cot_user, cot_assistant,
)

HERE = Path(__file__).resolve().parent

LIKERT_LABELS = ["Strongly agree", "Agree", "Uncertain", "Disagree", "Strongly disagree"]
YESNO_LABELS = ["Yes", "No"]


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--year", type=int, help="Single year to generate.")
    g.add_argument("--years", help="Range/list, e.g. 1931-2019 or 1935,1940.")
    p.add_argument("--config", type=Path, default=HERE / "config.yaml")
    p.add_argument("--corpus-root", type=Path, default=None,
                   help="Base dir holding per-year <Y>/legal.parquet (and where "
                        "<Y>/sft_corpus.jsonl is written). Matches run_synth_batch's "
                        "--corpus-root (e.g. C:/tmp/policy_pred/years). Default: "
                        "$POLICY_PRED_DATA_ROOT/years.")
    p.add_argument("--legal-path", type=Path, default=None,
                   help="Override the per-year legal parquet for a single --year.")
    p.add_argument("--out", type=Path, default=None,
                   help="Override output path (single --year only). "
                        "Default: <corpus-root>/<Y>/sft_corpus.jsonl.")
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cot", action="store_true",
                   help="Generate-once-format-twice: write a rich record "
                        "(question/reasoning/answer) plus BOTH renderings — "
                        "<Y>/sft_direct.jsonl (question->answer) and "
                        "<Y>/sft_cot.jsonl (question->reasoning->answer) — so the "
                        "two SFT variants differ ONLY in reasoning. Without it, "
                        "writes the legacy <Y>/sft_corpus.jsonl (direct only).")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def load_config(path: Path, args: argparse.Namespace) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for k in ("n_seeds", "model", "temperature", "max_workers", "seed"):
        v = getattr(args, k if k != "n_seeds" else "n_seeds", None)
        if v is not None:
            cfg[k] = v
    return cfg


def parse_years(spec: str) -> list[int]:
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = chunk.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def _years_root(corpus_root: Path | None) -> Path:
    """Base dir holding per-year <Y>/ subdirs (corpus-root, else data-root/years)."""
    if corpus_root is not None:
        return corpus_root
    root = Path(os.environ.get("POLICY_PRED_DATA_ROOT") or "D:/hist_LLM/policy_pred")
    return root / "years"


def _legal_path_for(year: int, override: Path | None, corpus_root: Path | None) -> Path:
    if override is not None:
        return override
    return _years_root(corpus_root) / str(year) / "legal.parquet"


def load_seed_passages(year: int, legal_path: Path, n_seeds: int, seed: int) -> list[str]:
    """First `n_seeds` passages in the canonical (stable_order) order.

    Sharing windowing.stable_order with the LoRA pool builder guarantees these
    passages are a prefix of the LoRA seed set.
    """
    import pandas as pd
    df = pd.read_parquet(legal_path)
    col = next((c for c in ("text", "text_cleaned", "combined_text", "article",
                            "cleaned_article") if c in df.columns), None)
    if col is None:
        raise ValueError(f"{legal_path} has no recognized text column")
    passages = [t for t in df[col].tolist() if isinstance(t, str) and t.strip()]
    order = windowing.stable_order(year, len(passages), seed=seed)
    return [passages[i] for i in order[:n_seeds]]


def load_banned_phrases(csv_path: Path) -> set[str]:
    """Distinctive catalog policy/case names to screen out of SFT questions.

    The SFT must not supply belief about the very policies the eval measures, so
    we drop any generated question that names a catalog policy/case. This catches
    blatant leakage (verbatim names); semantic topic-overlap is checked by the
    pre-scale audit, not here.
    """
    import csv
    import re as _re
    banned: set[str] = set()
    if not csv_path.exists():
        return banned
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        name = (r.get("policy_or_case") or "").strip().lower()
        if not name:
            continue
        banned.add(name)
        banned.add(_re.sub(r"\s+of\s+\d{4}$", "", name).strip())  # drop trailing year
        if " v. " in name:                                        # case party names
            banned.update(p.strip() for p in name.split(" v. "))
    return {b for b in banned if len(b) >= 5}                     # skip too-generic stubs


def call_one_seed(client, *, system: str, user_template: str, passage: str,
                  year: int, banned: set[str], model: str, temperature: float,
                  max_tokens: int, max_seed_chars: int) -> tuple[str, dict | None]:
    """One passage -> ("ok", rec) | ("leak", None) | ("fail", None).

    Single-direction: the question is the broad policy principle the passage
    instantiates, phrased one way; the answer follows the era's stance. "leak"
    means the generated question named a catalog policy/case (dropped).
    """
    user = user_template.replace("{year}", str(year)) \
                        .replace("{passage}", passage[:max_seed_chars])
    sys_prompt = system.replace("{year}", str(year))
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        rec = json.loads(_strip_code_fences(resp.choices[0].message.content))
    except Exception as e:  # API or JSON failure — skip this seed
        print(f"  [seed] error: {e}", file=sys.stderr)
        return "fail", None
    q = rec.get("question", "")
    yn = rec.get("yesno", "")
    lk = rec.get("likert", "")
    reasoning = str(rec.get("reasoning", "")).strip()
    if not (isinstance(q, str) and q.strip().endswith("?")
            and yn in YESNO_LABELS and lk in LIKERT_LABELS and reasoning):
        return "fail", None
    # Leakage screen covers BOTH the question and the reasoning chain: the CoT
    # reasoning is trained on verbatim, so a catalog name appearing there leaks
    # belief about the very policy the eval measures, just like one in the question.
    blob = f"{q}\n{reasoning}".lower()
    if any(b in blob for b in banned):
        return "leak", None
    return "ok", {"question": q.strip(), "yesno": yn, "likert": lk,
                  "reasoning": reasoning}


def rich_to_records(rich: dict, variant: str, year: int) -> list[dict]:
    """Render one rich record into two SFT chat records (yes/no + likert).

    The whole point of generate-once-format-twice: a single OpenAI call yields
    {question, reasoning, yesno, likert}; this renders it deterministically into
    whichever variant we want, so direct and CoT differ ONLY in the reasoning.

      variant="direct": question -> answer            (reasoning dropped)
      variant="cot":    question -> reasoning -> answer

    Both reuse the shared policy_battery builders, so each record is byte-identical
    to the eval prompt it will be scored against, up to the answer position.
    """
    q, yn, lk = rich["question"], rich["yesno"], rich["likert"]
    meta = {"generator": "sft_corpus", "year": year, "variant": variant,
            "yesno": yn, "likert": lk}
    if variant == "cot":
        r = rich["reasoning"]
        yn_user, yn_asst = build_yesno_cot_user(q), cot_assistant(r, yn)
        lk_user, lk_asst = build_likert_cot_user(q), cot_assistant(r, lk)
    else:
        yn_user, yn_asst = _build_yesno_prompt(q), yn  # content is .strip()'d at render
        lk_user, lk_asst = _build_likert_prompt(q), lk
    return [
        {"messages": [{"role": "user", "content": yn_user},
                      {"role": "assistant", "content": yn_asst}],
         "metadata": {**meta, "probe": "yes_no", "target_label": yn}},
        {"messages": [{"role": "user", "content": lk_user},
                      {"role": "assistant", "content": lk_asst}],
         "metadata": {**meta, "probe": "likert", "target_label": lk}},
    ]


def _write_records(rich_records: list[dict], variant: str, year: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rich in rich_records:
            for rec in rich_to_records(rich, variant, year):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"  wrote {n} {variant} records -> {path}")


def generate_year(client, year: int, cfg: dict, legal_path: Path, *,
                  cot: bool, out_dir: Path, out_override: Path | None,
                  force: bool) -> None:
    # CoT mode emits the rich record + both renderings; legacy mode emits the
    # single direct file (back-compat with the existing pod runs / build script).
    if cot:
        rich_path = out_dir / "sft_corpus.rich.jsonl"
        targets = {"rich": rich_path,
                   "direct": out_dir / "sft_direct.jsonl",
                   "cot": out_dir / "sft_cot.jsonl"}
        sentinel = rich_path
    else:
        sentinel = out_override or (out_dir / "sft_corpus.jsonl")
        targets = {"direct": sentinel}
    if sentinel.exists() and not force:
        print(f"  {sentinel} exists; --force to regenerate. Skip.")
        return

    n_seeds = int(cfg["n_seeds"])
    model = cfg.get("model", "gpt-4o-mini")
    temperature = float(cfg.get("temperature", 0.7))
    max_tokens = int(cfg.get("max_tokens", 600))
    max_workers = int(cfg.get("max_workers", 8))
    max_seed_chars = int(cfg.get("max_seed_chars", 4000))
    seed = int(cfg.get("seed", 0))

    passages = load_seed_passages(year, legal_path, n_seeds, seed)
    system = (HERE / "prompts" / "system.md").read_text(encoding="utf-8")
    user_template = (HERE / "prompts" / "user.md").read_text(encoding="utf-8")
    banned = load_banned_phrases(Path(cfg.get("battery_csv", "us_policy_event_battery_v4.csv")))
    print(f"  year {year}: {len(passages)} seed passages, model={model}, "
          f"{len(banned)} catalog phrases screened, cot={cot}")

    rich_records: list[dict] = []
    label_hist: dict[str, int] = {}
    n_ok = n_leak = n_fail = done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(call_one_seed, client, system=system,
                             user_template=user_template, passage=p, year=year,
                             banned=banned, model=model, temperature=temperature,
                             max_tokens=max_tokens, max_seed_chars=max_seed_chars)
                   for p in passages]
        for fut in as_completed(futures):
            status, res = fut.result()
            done += 1
            if status == "leak":
                n_leak += 1
            elif status == "fail":
                n_fail += 1
            else:
                n_ok += 1
                rich_records.append(res)
                label_hist[res["yesno"]] = label_hist.get(res["yesno"], 0) + 1
                label_hist[res["likert"]] = label_hist.get(res["likert"], 0) + 1
            if done % 100 == 0 or done == len(passages):
                print(f"    {done}/{len(passages)} seeds, {n_ok} ok, "
                      f"{time.time()-t0:.0f}s")

    # Persist the rich record (CoT mode) so the two renderings are auditable +
    # re-formattable without paying OpenAI again, then write each rendering.
    if "rich" in targets:
        targets["rich"].parent.mkdir(parents=True, exist_ok=True)
        with open(targets["rich"], "w", encoding="utf-8") as f:
            for rich in rich_records:
                f.write(json.dumps(rich, ensure_ascii=False) + "\n")
        print(f"  wrote {len(rich_records)} rich records -> {targets['rich']}")
    for variant in ("direct", "cot"):
        if variant in targets:
            _write_records(rich_records, variant, year, targets[variant])
    print(f"  ok={n_ok}, leak-dropped={n_leak}, fail={n_fail}")
    print("  label distribution (NOT forced — emerges from corpus):")
    for label in YESNO_LABELS + LIKERT_LABELS:
        print(f"    {label:>16s}: {label_hist.get(label, 0)}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args)
    years = [args.year] if args.year is not None else parse_years(args.years)
    if args.out is not None and len(years) != 1:
        sys.exit("--out is only valid with a single --year")
    if args.out is not None and args.cot:
        sys.exit("--out is not valid with --cot (it writes three named files into <Y>/)")

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("Set OPENAI_API_KEY env var first")
    from openai import OpenAI
    client = OpenAI()

    years_root = _years_root(args.corpus_root)
    for year in years:
        legal_path = _legal_path_for(year, args.legal_path, args.corpus_root)
        if not legal_path.exists():
            print(f"  year {year}: legal corpus missing at {legal_path}; skip.")
            continue
        print(f"=== {year} ===")
        generate_year(client, year, cfg, legal_path, cot=args.cot,
                      out_dir=years_root / str(year), out_override=args.out,
                      force=args.force)


if __name__ == "__main__":
    main()
