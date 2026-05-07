"""synth_2step: Anthropic-style two-stage synthetic-data generation.

For each seed document, runs two stages:
  1. IDEATE -- one OpenAI call -> list of N {doc_type, concept, tone} records.
  2. WRITE  -- one OpenAI call per idea -> full synthetic document.

Total per seed: 1 + N API calls. Output is JSONL with {"text": "..."} records.

Inspired by Anthropic's Model Spec Midtraining repo
(github.com/chloeli-15/model_spec_midtraining), with their hierarchical
domain/subdomain decomposition stripped -- we use the seed text itself as
the diversity source.

Out of the box (no args), reads examples/seeds.txt and writes to ./out/.

Usage:
    python run.py                                    # use defaults
    python run.py --seeds my_corpus.parquet --out ./run1
    python run.py --model gpt-4o --n-ideas 12 --limit 5

Override anything in config.yaml via CLI flags.

Env: OPENAI_API_KEY must be set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# Config + CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=HERE / "config.yaml")
    p.add_argument("--seeds", type=Path, default=None,
                   help="Override config.seeds. .txt/.jsonl/.parquet/dir.")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--prompts", type=Path, default=None,
                   help="Override prompts dir (must contain ideas.md + writer.md).")
    p.add_argument("--model", default=None)
    p.add_argument("--n-ideas", type=int, default=None,
                   help="Doc ideas per seed = docs per seed.")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens-ideas", type=int, default=None)
    p.add_argument("--max-tokens-writer", type=int, default=None)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--banned-terms", type=Path, default=None,
                   help="One banned substring per line, case-insensitive. "
                        "Outputs containing any banned term are dropped.")
    return p.parse_args()


def load_config(config_path: Path, args: argparse.Namespace) -> dict:
    import yaml
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    overrides = {
        "seeds": args.seeds,
        "out": args.out,
        "prompts": args.prompts,
        "model": args.model,
        "n_ideas": args.n_ideas,
        "temperature": args.temperature,
        "max_tokens_ideas": args.max_tokens_ideas,
        "max_tokens_writer": args.max_tokens_writer,
        "max_workers": args.max_workers,
        "limit": args.limit,
        "banned_terms": args.banned_terms,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    for path_key in ("seeds", "out", "prompts", "banned_terms"):
        v = cfg.get(path_key)
        if v is not None and not Path(v).is_absolute():
            cfg[path_key] = HERE / v
    return cfg


# ----------------------------------------------------------------------------
# Seed loading (auto-detect format)
# ----------------------------------------------------------------------------

def load_seeds(path: Path) -> list[str]:
    if path.is_dir():
        return [f.read_text(encoding="utf-8").strip()
                for f in sorted(path.glob("*.txt"))
                if f.read_text(encoding="utf-8").strip()]
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8")
        if "\n---\n" in text:
            return [s.strip() for s in text.split("\n---\n") if s.strip()]
        return [line.strip() for line in text.splitlines() if line.strip()]
    if suffix == ".jsonl":
        seeds = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                t = rec.get("text") or rec.get("content") or rec.get("body")
                if isinstance(t, str) and t.strip():
                    seeds.append(t.strip())
        return seeds
    if suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        for col in ("text", "text_cleaned", "combined_text", "article", "cleaned_article"):
            if col in df.columns:
                return [t for t in df[col].dropna().tolist()
                        if isinstance(t, str) and t.strip()]
        raise ValueError(f"parquet at {path} has no recognized text column")
    raise ValueError(f"unsupported seeds format: {path.suffix} for {path}")


# ----------------------------------------------------------------------------
# OpenAI calls
# ----------------------------------------------------------------------------

def _chat(client, system: str, user: str, *,
          model: str, temperature: float, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def parse_ideas(content: str) -> list[dict]:
    """Pull idea records out of stage-1 JSON. Returns list of {doc_type, concept, tone}."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return []
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        items = obj.get("ideas") or obj.get("items") or []
    else:
        return []
    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if "doc_type" not in it or "concept" not in it:
            continue
        out.append({
            "doc_type": str(it["doc_type"]).strip(),
            "concept": str(it["concept"]).strip(),
            "tone": str(it.get("tone", "")).strip(),
        })
    return out


def parse_document(content: str) -> str | None:
    """Pull the document string out of stage-2 JSON."""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict):
        text = obj.get("document") or obj.get("text") or obj.get("content")
        if isinstance(text, str) and text.strip():
            return text.strip()
    if isinstance(obj, str):
        return obj.strip() or None
    return None


# ----------------------------------------------------------------------------
# Leakage filter
# ----------------------------------------------------------------------------

def load_banned_terms(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def has_banned_term(text: str, banned: list[str]) -> str | None:
    if not banned:
        return None
    lower = text.lower()
    for term in banned:
        if term in lower:
            return term
    return None


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args)

    seeds_path = Path(cfg["seeds"])
    out_dir = Path(cfg["out"])
    prompts_dir = Path(cfg.get("prompts") or HERE / "prompts")
    banned_path = Path(cfg["banned_terms"]) if cfg.get("banned_terms") else None

    if not seeds_path.exists():
        sys.exit(f"seeds not found: {seeds_path}")
    if not prompts_dir.is_dir():
        sys.exit(f"prompts directory not found: {prompts_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "log"
    log_dir.mkdir(exist_ok=True)

    ideas_prompt = (prompts_dir / "ideas.md").read_text(encoding="utf-8")
    writer_prompt = (prompts_dir / "writer.md").read_text(encoding="utf-8")
    seeds = load_seeds(seeds_path)
    if not seeds:
        sys.exit(f"no seeds at {seeds_path}")
    if cfg.get("limit"):
        seeds = seeds[: int(cfg["limit"])]
    banned = load_banned_terms(banned_path)

    n_ideas = int(cfg["n_ideas"])
    print(f"synth_2step  |  seeds={seeds_path}  ({len(seeds)} docs)  ->  out={out_dir}")
    print(f"  model={cfg['model']}  n_ideas={n_ideas}  workers={cfg.get('max_workers', 8)}")
    print(f"  total API calls: {len(seeds)} (ideate) + {len(seeds) * n_ideas} (write)")
    if banned:
        print(f"  leakage filter: {len(banned)} banned terms loaded from {banned_path}")

    # Snapshot config.
    import yaml
    with open(log_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump({k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
                  f, sort_keys=True)

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("OPENAI_API_KEY env var not set")
    from openai import OpenAI
    client = OpenAI()

    ideas_path = out_dir / "ideas.jsonl"
    out_path = out_dir / "synth.jsonl"
    failures_path = log_dir / "failures.jsonl"

    n_ideas_total = 0
    n_records = 0
    n_dropped = 0
    n_failures = 0
    t0 = time.time()

    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:
        def tqdm(it, **_):
            return it

    # ---- Stage 1: ideate (one call per seed) ------------------------------
    print("\nStage 1: generating ideas...")

    def _ideate(idx: int, seed_text: str):
        seed_truncated = seed_text[: int(cfg.get("max_seed_chars", 4000))]
        user = ideas_prompt.format(seed=seed_truncated, n_ideas=n_ideas) \
            if "{seed}" in ideas_prompt else \
            f"Seed text:\n\n```\n{seed_truncated}\n```\n\nGenerate {n_ideas} doc-idea records as specified."
        try:
            content = _chat(client, ideas_prompt, user,
                            model=cfg["model"],
                            temperature=float(cfg["temperature"]),
                            max_tokens=int(cfg["max_tokens_ideas"]))
            return idx, parse_ideas(content), None
        except Exception as e:  # noqa: BLE001
            return idx, [], e

    seed_to_ideas: dict[int, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=int(cfg.get("max_workers", 8))) as ex:
        futures = {ex.submit(_ideate, i, s): i for i, s in enumerate(seeds)}
        with open(ideas_path, "w", encoding="utf-8") as ideas_f, \
             open(failures_path, "w", encoding="utf-8") as fail_f:
            for fut in tqdm(as_completed(futures), total=len(futures), desc="ideate"):
                idx, ideas, err = fut.result()
                if err is not None:
                    n_failures += 1
                    fail_f.write(json.dumps({"stage": "ideate", "seed_idx": idx,
                                             "error": str(err), "type": type(err).__name__}) + "\n")
                    continue
                seed_to_ideas[idx] = ideas
                for j, idea in enumerate(ideas):
                    ideas_f.write(json.dumps({
                        "seed_idx": idx, "idea_idx": j, **idea
                    }, ensure_ascii=False) + "\n")
                    n_ideas_total += 1

    print(f"  -> {n_ideas_total} ideas across {len(seed_to_ideas)} seeds")

    # ---- Stage 2: write (one call per idea) -------------------------------
    print(f"\nStage 2: writing documents ({n_ideas_total} calls)...")

    def _write(seed_idx: int, idea_idx: int, seed_text: str, idea: dict):
        seed_truncated = seed_text[: int(cfg.get("max_seed_chars", 4000))]
        user = (
            f"Seed era context (do not reference directly):\n```\n{seed_truncated}\n```\n\n"
            f"Doc idea:\n"
            f"  doc_type: {idea['doc_type']}\n"
            f"  concept: {idea['concept']}\n"
            f"  tone: {idea.get('tone', '')}\n\n"
            f"Write the full document per the rules in the system prompt. "
            f"Return JSON {{\"document\": \"...\"}}."
        )
        try:
            content = _chat(client, writer_prompt, user,
                            model=cfg["model"],
                            temperature=float(cfg["temperature"]),
                            max_tokens=int(cfg["max_tokens_writer"]))
            return seed_idx, idea_idx, parse_document(content), None
        except Exception as e:  # noqa: BLE001
            return seed_idx, idea_idx, None, e

    write_jobs = []
    for seed_idx, ideas in seed_to_ideas.items():
        for j, idea in enumerate(ideas):
            write_jobs.append((seed_idx, j, seeds[seed_idx], idea))

    with ThreadPoolExecutor(max_workers=int(cfg.get("max_workers", 8))) as ex, \
         open(out_path, "w", encoding="utf-8") as out_f, \
         open(failures_path, "a", encoding="utf-8") as fail_f:
        futures = {ex.submit(_write, *args): args for args in write_jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="write"):
            seed_idx, idea_idx, text, err = fut.result()
            args_tuple = futures[fut]
            idea = args_tuple[3]
            if err is not None:
                n_failures += 1
                fail_f.write(json.dumps({
                    "stage": "write", "seed_idx": seed_idx, "idea_idx": idea_idx,
                    "error": str(err), "type": type(err).__name__,
                }) + "\n")
                continue
            if text is None:
                n_failures += 1
                fail_f.write(json.dumps({
                    "stage": "write", "seed_idx": seed_idx, "idea_idx": idea_idx,
                    "error": "parse_failed",
                }) + "\n")
                continue
            bad = has_banned_term(text, banned)
            if bad is not None:
                n_dropped += 1
                fail_f.write(json.dumps({
                    "stage": "write", "seed_idx": seed_idx, "idea_idx": idea_idx,
                    "reason": "banned_term", "term": bad,
                }) + "\n")
                continue
            out_f.write(json.dumps({
                "text": text,
                "metadata": {
                    "seed_idx": seed_idx,
                    "idea_idx": idea_idx,
                    "doc_type": idea["doc_type"],
                    "tone": idea.get("tone", ""),
                    "generator": "synth_2step",
                },
            }, ensure_ascii=False) + "\n")
            n_records += 1

    elapsed = time.time() - t0
    print()
    print(f"  ideas generated:  {n_ideas_total}")
    print(f"  documents written: {n_records}")
    print(f"  failures:         {n_failures}")
    if banned:
        print(f"  dropped by filter: {n_dropped}")
    print(f"  elapsed:          {elapsed/60:.1f} min")
    print(f"  ideas log:        {ideas_path}")
    print(f"  output:           {out_path}")


if __name__ == "__main__":
    main()
