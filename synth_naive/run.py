"""synth_naive: one-step synthetic-data generation from seed documents.

For each seed document, makes one OpenAI call asking for N synthetic
training documents inspired by the seed. Output is JSONL with
{"text": "..."} records, one per generated document.

Out of the box (no args), reads examples/seeds.txt and writes to ./out/.

Usage:
    python run.py                                          # use defaults
    python run.py --seeds my_corpus.parquet --out ./run1
    python run.py --model gpt-4o --n-per-seed 5 --limit 10

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
    p.add_argument("--config", type=Path, default=HERE / "config.yaml",
                   help="Path to config.yaml (default: bundled).")
    p.add_argument("--seeds", type=Path, default=None,
                   help="Override config.seeds. Path to .txt/.jsonl/.parquet/dir.")
    p.add_argument("--out", type=Path, default=None,
                   help="Override config.out. Output directory.")
    p.add_argument("--prompts", type=Path, default=None,
                   help="Override prompts directory (must contain system.md + user.md).")
    p.add_argument("--model", default=None, help="Override OpenAI model.")
    p.add_argument("--n-per-seed", type=int, default=None,
                   help="Override n_per_seed.")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of seeds processed (smoke test).")
    p.add_argument("--banned-terms", type=Path, default=None,
                   help="Path to a .txt file of banned substrings (one per line). "
                        "Output records containing any banned term are dropped.")
    return p.parse_args()


def load_config(config_path: Path, args: argparse.Namespace) -> dict:
    """Load config.yaml and merge CLI overrides on top."""
    import yaml
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # CLI > config
    overrides = {
        "seeds": args.seeds,
        "out": args.out,
        "prompts": args.prompts,
        "model": args.model,
        "n_per_seed": args.n_per_seed,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_workers": args.max_workers,
        "limit": args.limit,
        "banned_terms": args.banned_terms,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    # Resolve relative paths against the synth_naive folder.
    for path_key in ("seeds", "out", "prompts", "banned_terms"):
        v = cfg.get(path_key)
        if v is not None and not Path(v).is_absolute():
            cfg[path_key] = HERE / v
    return cfg


# ----------------------------------------------------------------------------
# Seed loading (auto-detect format)
# ----------------------------------------------------------------------------

def load_seeds(path: Path) -> list[str]:
    """Read seed documents from path. Returns list[str]; one entry per seed."""
    if path.is_dir():
        seeds = []
        for f in sorted(path.glob("*.txt")):
            text = f.read_text(encoding="utf-8").strip()
            if text:
                seeds.append(text)
        return seeds

    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8")
        # Prefer "---" separator; fall back to one-doc-per-line if none.
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
        raise ValueError(
            f"parquet at {path} has no recognized text column "
            f"(tried text, text_cleaned, combined_text, article, cleaned_article)"
        )

    raise ValueError(f"unsupported seeds format: {path.suffix} for {path}")


# ----------------------------------------------------------------------------
# OpenAI call + parsing
# ----------------------------------------------------------------------------

def render_prompt(template: str, **kwargs) -> str:
    return template.format(**kwargs)


def call_openai(client, system: str, user: str, *,
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


def parse_documents(content: str) -> list[str]:
    """Pull document strings out of a JSON response.

    Accepts {"documents": [...]} (preferred) or a bare list. Rejects malformed
    or empty results.
    """
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return []
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        items = obj.get("documents") or obj.get("docs") or obj.get("items") or []
    else:
        return []
    out: list[str] = []
    for it in items:
        if isinstance(it, str) and it.strip():
            out.append(it.strip())
        elif isinstance(it, dict) and isinstance(it.get("text"), str):
            out.append(it["text"].strip())
    return out


# ----------------------------------------------------------------------------
# Leakage filter (optional)
# ----------------------------------------------------------------------------

def load_banned_terms(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def has_banned_term(text: str, banned: list[str]) -> str | None:
    """Return the first banned term found, or None."""
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

    # Resolve paths.
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

    # Load resources.
    system_prompt = (prompts_dir / "system.md").read_text(encoding="utf-8")
    user_template = (prompts_dir / "user.md").read_text(encoding="utf-8")
    seeds = load_seeds(seeds_path)
    if not seeds:
        sys.exit(f"no seeds found at {seeds_path}")
    if cfg.get("limit"):
        seeds = seeds[: int(cfg["limit"])]
    banned = load_banned_terms(banned_path)

    print(f"synth_naive  |  seeds={seeds_path}  ({len(seeds)} docs)  ->  out={out_dir}")
    print(f"  model={cfg['model']}  n_per_seed={cfg['n_per_seed']}  workers={cfg.get('max_workers', 8)}")
    if banned:
        print(f"  leakage filter: {len(banned)} banned terms loaded from {banned_path}")

    # Snapshot effective config.
    import yaml
    with open(log_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump({k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
                  f, sort_keys=True)

    # API client.
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("OPENAI_API_KEY env var not set")
    from openai import OpenAI
    client = OpenAI()

    out_path = out_dir / "synth.jsonl"
    failures_path = log_dir / "failures.jsonl"

    n_records = 0
    n_dropped_by_filter = 0
    n_failures = 0
    t0 = time.time()

    def _process_one(idx: int, seed_text: str) -> tuple[int, list[str], Exception | None]:
        seed_truncated = seed_text[: int(cfg.get("max_seed_chars", 4000))]
        user = render_prompt(user_template,
                             seed=seed_truncated,
                             n_per_seed=int(cfg["n_per_seed"]))
        try:
            content = call_openai(client, system_prompt, user,
                                  model=cfg["model"],
                                  temperature=float(cfg["temperature"]),
                                  max_tokens=int(cfg["max_tokens"]))
            return idx, parse_documents(content), None
        except Exception as e:  # noqa: BLE001
            return idx, [], e

    try:
        from tqdm import tqdm  # type: ignore
    except ImportError:  # pragma: no cover
        def tqdm(it, **_):
            return it

    with open(out_path, "w", encoding="utf-8") as out_f, \
         open(failures_path, "w", encoding="utf-8") as fail_f, \
         ThreadPoolExecutor(max_workers=int(cfg.get("max_workers", 8))) as ex:
        futures = {ex.submit(_process_one, i, s): i for i, s in enumerate(seeds)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="seeds"):
            idx, docs, err = fut.result()
            if err is not None:
                n_failures += 1
                fail_f.write(json.dumps({
                    "seed_idx": idx, "error": str(err), "type": type(err).__name__
                }) + "\n")
                continue
            for d in docs:
                bad = has_banned_term(d, banned)
                if bad is not None:
                    n_dropped_by_filter += 1
                    fail_f.write(json.dumps({
                        "seed_idx": idx, "reason": "banned_term", "term": bad
                    }) + "\n")
                    continue
                out_f.write(json.dumps({
                    "text": d,
                    "metadata": {"seed_idx": idx, "generator": "synth_naive"}
                }, ensure_ascii=False) + "\n")
                n_records += 1

    elapsed = time.time() - t0
    print()
    print(f"  generated: {n_records} records")
    print(f"  failures:  {n_failures}")
    if banned:
        print(f"  dropped by filter: {n_dropped_by_filter}")
    print(f"  elapsed:   {elapsed/60:.1f} min")
    print(f"  output:    {out_path}")


if __name__ == "__main__":
    main()
