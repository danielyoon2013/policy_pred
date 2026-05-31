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

HERE = Path(__file__).resolve().parent

LIKERT_LABELS = ["Strongly agree", "Agree", "Uncertain", "Disagree", "Strongly disagree"]
YESNO_LABELS = ["Yes", "No"]


# --- Probe prompt formatting — must MATCH evaluators/policy_battery.py exactly ---

def _build_yesno_user_prompt(question: str) -> str:
    return f"{question}\n\nAnswer:"


def _build_likert_user_prompt(question: str) -> str:
    return ("Respond to the following question by rating your agreement:\n"
            "\n"
            f"{question}\n"
            "\n"
            "Answer:")


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
    p.add_argument("--legal-path", type=Path, default=None,
                   help="Override the per-year legal parquet "
                        "(default: $POLICY_PRED_DATA_ROOT/years/<Y>/legal.parquet).")
    p.add_argument("--out", type=Path, default=None,
                   help="Override output path (single --year only). "
                        "Default: <data_root>/years/<Y>/sft_corpus.jsonl.")
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
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


def _legal_path_for(year: int, override: Path | None) -> Path:
    if override is not None:
        return override
    root = Path(os.environ.get("POLICY_PRED_DATA_ROOT") or "D:/hist_LLM/policy_pred")
    return root / "years" / str(year) / "legal.parquet"


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


def call_one_seed(client, *, system: str, user_template: str, passage: str,
                  year: int, frame: str, model: str, temperature: float,
                  max_tokens: int, max_seed_chars: int) -> dict | None:
    """One passage -> validated {question, yesno, likert, rationale} or None.

    `frame` ("affirm" | "oppose") balances question polarity: legal holdings are
    overwhelmingly affirmative, so without alternating the frame the labels skew
    ~90% Yes / Strongly-agree. Alternating yields a natural spread while each
    label stays grounded in the passage's actual stance.
    """
    user = user_template.replace("{year}", str(year)) \
                        .replace("{frame}", frame) \
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
        return None
    q = rec.get("question", "")
    yn = rec.get("yesno", "")
    lk = rec.get("likert", "")
    if not (isinstance(q, str) and q.strip().endswith("?")
            and yn in YESNO_LABELS and lk in LIKERT_LABELS):
        return None
    return {"question": q.strip(), "yesno": yn, "likert": lk,
            "rationale": str(rec.get("rationale", "")).strip()}


def generate_year(client, year: int, cfg: dict, legal_path: Path, out_path: Path,
                  force: bool) -> None:
    if out_path.exists() and not force:
        with open(out_path, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        print(f"  {out_path} exists ({n} records); --force to regenerate. Skip.")
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
    print(f"  year {year}: {len(passages)} seed passages, model={model}")

    records: list[dict] = []
    label_hist: dict[str, int] = {}
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Alternate affirm/oppose framing per passage to balance question polarity.
        futures = [ex.submit(call_one_seed, client, system=system,
                             user_template=user_template, passage=p, year=year,
                             frame=("affirm" if i % 2 == 0 else "oppose"),
                             model=model, temperature=temperature,
                             max_tokens=max_tokens, max_seed_chars=max_seed_chars)
                   for i, p in enumerate(passages)]
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            if res is not None:
                meta = {"generator": "sft_corpus", "year": year,
                        "yesno": res["yesno"], "likert": res["likert"]}
                records.append({
                    "messages": [
                        {"role": "user", "content": _build_yesno_user_prompt(res["question"])},
                        {"role": "assistant", "content": f" {res['yesno']}"}],
                    "metadata": {**meta, "probe": "yes_no", "target_label": res["yesno"]}})
                records.append({
                    "messages": [
                        {"role": "user", "content": _build_likert_user_prompt(res["question"])},
                        {"role": "assistant", "content": f" {res['likert']}"}],
                    "metadata": {**meta, "probe": "likert", "target_label": res["likert"]}})
                label_hist[res["yesno"]] = label_hist.get(res["yesno"], 0) + 1
                label_hist[res["likert"]] = label_hist.get(res["likert"], 0) + 1
            if done % 100 == 0 or done == len(passages):
                print(f"    {done}/{len(passages)} seeds, {len(records)} records, "
                      f"{time.time()-t0:.0f}s")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records)} records -> {out_path}")
    print("  label distribution (NOT forced — emerges from corpus):")
    for label in YESNO_LABELS + LIKERT_LABELS:
        print(f"    {label:>16s}: {label_hist.get(label, 0)}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args)
    years = [args.year] if args.year is not None else parse_years(args.years)
    if args.out is not None and len(years) != 1:
        sys.exit("--out is only valid with a single --year")

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("Set OPENAI_API_KEY env var first")
    from openai import OpenAI
    client = OpenAI()

    data_root = Path(os.environ.get("POLICY_PRED_DATA_ROOT") or "D:/hist_LLM/policy_pred")
    for year in years:
        legal_path = _legal_path_for(year, args.legal_path)
        if not legal_path.exists():
            print(f"  year {year}: legal corpus missing at {legal_path}; skip.")
            continue
        out_path = args.out or (data_root / "years" / str(year) / "sft_corpus.jsonl")
        print(f"=== {year} ===")
        generate_year(client, year, cfg, legal_path, out_path, args.force)


if __name__ == "__main__":
    main()
