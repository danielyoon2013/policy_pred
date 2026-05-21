"""synth/sft_format: generate format-teaching SFT pairs for the policy probe.

Produces {"messages":[user, assistant]} JSONL records in the EXACT shape
the policy battery probes use at eval time. Used as the SFT stage of the
policy column in the matrix-experiments plan — teaches the model HOW to
answer Yes/No and likert prompts without teaching it WHAT to say about
any specific policy.

Two output classes, balanced by label:
- 5K yes/no pairs: exactly 2,500 with assistant=" Yes", 2,500 with " No".
- 5K likert pairs: exactly 1,000 each across
  {Strongly agree, Agree, Uncertain, Disagree, Strongly disagree}.

Both classes are merged + shuffled into a single jsonl. Topics come from
gpt-4o-mini paraphrasing — we don't supply explicit seeds. Topic diversity
comes from the LLM. We just supply the target label per batch so the
output is correctly distributed.

Usage:
    cd synth/sft_format
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...
    python run.py                                # use config.yaml defaults
    python run.py --n-yesno 1000 --n-likert 1000 # smaller smoke run

Env: OPENAI_API_KEY must be set.
Idempotent: skips generation if --out/sft_format.jsonl already exists.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent

LIKERT_LABELS = [
    "Strongly agree",
    "Agree",
    "Uncertain",
    "Disagree",
    "Strongly disagree",
]


# ----------------------------------------------------------------------------
# Config + CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=HERE / "config.yaml")
    p.add_argument("--out", type=Path, default=None,
                   help="Override config.out. Default: ./out/sft_format.jsonl.")
    p.add_argument("--n-yesno", type=int, default=None,
                   help="Total yes/no pairs (balanced 50/50). Override config.")
    p.add_argument("--n-likert", type=int, default=None,
                   help="Total likert pairs (balanced 5-way). Override config.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Questions per OpenAI call. Lower = more API calls "
                        "but smaller blast radius if any call fails.")
    p.add_argument("--model", default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for label-batch ordering + final shuffle.")
    p.add_argument("--force", action="store_true",
                   help="Regenerate even if --out exists.")
    return p.parse_args()


def load_config(config_path: Path, args: argparse.Namespace) -> dict:
    import yaml
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    overrides = {
        "out": args.out,
        "n_yesno": args.n_yesno,
        "n_likert": args.n_likert,
        "batch_size": args.batch_size,
        "model": args.model,
        "temperature": args.temperature,
        "max_workers": args.max_workers,
        "seed": args.seed,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


# ----------------------------------------------------------------------------
# OpenAI call
# ----------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """gpt-4o-mini sometimes wraps JSON in ```json ... ``` fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


def call_one_batch(
    client, *, template: str, target_label: str, n_pairs: int,
    model: str, temperature: float, max_tokens: int,
) -> list[dict]:
    """One API call → list of {"question": ...} dicts.

    Returns up to n_pairs items. Empty list on parse/API failure; caller
    decides whether to retry or skip.
    """
    prompt = template.replace("{n_pairs}", str(n_pairs)) \
                     .replace("{target_label}", target_label)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content
    except Exception as e:
        print(f"  [call_one_batch] API error for label={target_label!r}: {e}",
              file=sys.stderr)
        return []
    try:
        items = json.loads(_strip_code_fences(content))
    except json.JSONDecodeError as e:
        print(f"  [call_one_batch] JSON parse error for label={target_label!r}:"
              f" {e}; got: {content[:200]!r}", file=sys.stderr)
        return []
    if not isinstance(items, list):
        return []
    # Defensive: keep only items with a non-empty "question" field
    out = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("question"), str):
            q = it["question"].strip()
            if q.endswith("?"):
                out.append({"question": q})
    return out


# ----------------------------------------------------------------------------
# Probe prompt formatting — must MATCH evaluators/policy_battery.py exactly
# ----------------------------------------------------------------------------

def _build_yesno_user_prompt(question: str) -> str:
    """Matches evaluators/policy_battery.py:_build_yesno_prompt."""
    return f"{question}\n\nAnswer:"


def _build_likert_user_prompt(question: str) -> str:
    """Matches evaluators/policy_battery.py:_build_likert_prompt."""
    return (
        "Respond to the following question by rating your agreement:\n"
        "\n"
        f"{question}\n"
        "\n"
        "Answer:"
    )


# ----------------------------------------------------------------------------
# Pair pipeline
# ----------------------------------------------------------------------------

def _split_into_batches(total: int, batch_size: int) -> list[int]:
    """Split `total` into a list of batch sizes summing to total."""
    full, rem = divmod(total, batch_size)
    return [batch_size] * full + ([rem] if rem else [])


def generate_class(
    client,
    *,
    template_path: Path,
    labels: list[str],
    counts: dict[str, int],
    batch_size: int,
    model: str,
    temperature: float,
    max_tokens: int,
    max_workers: int,
    user_prompt_fn,
) -> list[dict]:
    """Drive a balanced generation across `labels` with target `counts[label]`.

    Returns the merged list of {"messages": [...]} records (untrimmed —
    callers should trim to exact counts before write).
    """
    template = template_path.read_text(encoding="utf-8")

    # Build a flat list of (label, batch_n) jobs
    jobs: list[tuple[str, int]] = []
    for label in labels:
        for batch_n in _split_into_batches(counts[label], batch_size):
            jobs.append((label, batch_n))

    records: list[dict] = []
    t0 = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                call_one_batch,
                client,
                template=template,
                target_label=label,
                n_pairs=n,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            ): (label, n)
            for label, n in jobs
        }
        for fut in as_completed(futures):
            label, n_requested = futures[fut]
            items = fut.result()
            for it in items:
                records.append({
                    "messages": [
                        {"role": "user", "content": user_prompt_fn(it["question"])},
                        {"role": "assistant", "content": f" {label}"},
                    ],
                    "metadata": {"target_label": label, "generator": "sft_format"},
                })
            completed += 1
            if completed % 10 == 0 or completed == len(jobs):
                elapsed = time.time() - t0
                print(f"  progress: {completed}/{len(jobs)} batches "
                      f"({len(records)} records so far, {elapsed:.0f}s)")
    return records


def trim_to_counts(records: list[dict], targets: dict[str, int]) -> list[dict]:
    """Sample-or-truncate down to the exact per-label targets.

    The LLM may over- or under-deliver per batch; this function enforces
    the final 50/50 (yes/no) or 20%-each (likert) balance the plan demands.
    """
    by_label: dict[str, list[dict]] = {}
    for r in records:
        by_label.setdefault(r["metadata"]["target_label"], []).append(r)

    out: list[dict] = []
    for label, want in targets.items():
        pool = by_label.get(label, [])
        if len(pool) < want:
            print(f"  WARN: only {len(pool)} records for {label!r}, "
                  f"wanted {want}", file=sys.stderr)
            out.extend(pool)
        else:
            out.extend(pool[:want])
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args)

    out_path = Path(cfg.get("out", "./out/sft_format.jsonl"))
    if not out_path.is_absolute():
        out_path = HERE / out_path

    if out_path.exists() and not args.force:
        with open(out_path, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        print(f"{out_path} already exists ({n} records). Use --force to regenerate.")
        return

    n_yesno = int(cfg["n_yesno"])
    n_likert = int(cfg["n_likert"])
    batch_size = int(cfg.get("batch_size", 10))
    model = cfg.get("model", "gpt-4o-mini")
    temperature = float(cfg.get("temperature", 0.9))
    max_tokens = int(cfg.get("max_tokens", 2048))
    max_workers = int(cfg.get("max_workers", 16))
    seed = int(cfg.get("seed", 42))
    random.seed(seed)

    print(f"Output:       {out_path}")
    print(f"Target counts: {n_yesno} yes/no + {n_likert} likert "
          f"= {n_yesno + n_likert} total pairs")
    print(f"Model:        {model} @ temperature={temperature}")
    print(f"Batch size:   {batch_size} pairs per OpenAI call")
    print(f"Parallelism:  {max_workers} concurrent calls")
    print()

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("Set OPENAI_API_KEY env var first")
    from openai import OpenAI
    client = OpenAI()

    # === Yes/No phase (balanced 50/50) ===
    print("=== Phase 1: yes/no pairs ===")
    yesno_targets = {"Yes": n_yesno // 2, "No": n_yesno - n_yesno // 2}
    yesno_records = generate_class(
        client,
        template_path=HERE / "prompts" / "yesno_template.md",
        labels=list(yesno_targets.keys()),
        counts=yesno_targets,
        batch_size=batch_size,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_workers=max_workers,
        user_prompt_fn=_build_yesno_user_prompt,
    )
    yesno_records = trim_to_counts(yesno_records, yesno_targets)
    print(f"  yes/no: {len(yesno_records)} pairs after trim")

    # === Likert phase (balanced 5-way) ===
    print()
    print("=== Phase 2: likert pairs ===")
    per_label = n_likert // 5
    likert_targets = {label: per_label for label in LIKERT_LABELS}
    # Distribute remainder
    rem = n_likert - per_label * 5
    for i in range(rem):
        likert_targets[LIKERT_LABELS[i]] += 1
    likert_records = generate_class(
        client,
        template_path=HERE / "prompts" / "likert_template.md",
        labels=LIKERT_LABELS,
        counts=likert_targets,
        batch_size=batch_size,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_workers=max_workers,
        user_prompt_fn=_build_likert_user_prompt,
    )
    likert_records = trim_to_counts(likert_records, likert_targets)
    print(f"  likert: {len(likert_records)} pairs after trim")

    # === Merge + shuffle + write ===
    all_records = yesno_records + likert_records
    random.shuffle(all_records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote {len(all_records)} records -> {out_path}")
    # Quick sanity print: count by target_label
    counts: dict[str, int] = {}
    for r in all_records:
        counts[r["metadata"]["target_label"]] = counts.get(
            r["metadata"]["target_label"], 0
        ) + 1
    for label, n in sorted(counts.items()):
        print(f"  {label:>20s}: {n}")


if __name__ == "__main__":
    main()
