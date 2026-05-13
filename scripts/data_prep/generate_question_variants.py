"""Generate N LLM-paraphrased variants of each policy's benchmark_question.

For the multi-policy panel eval we want a per-variant mean+std of P(yes)
and likert score, so reviewers can see how sensitive each year-model's
belief is to wording. This script makes ~100 paraphrases per policy.

Reads `us_policy_event_battery_v4.csv`, sends each row's `benchmark_question`
to OpenAI gpt-4o-mini in batches of N/5 to stay within max_tokens, writes
output to `data_artifacts/question_variants/<event_id>.jsonl` with rows
`{event_id, variant_idx, variant_question}`.

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/data_prep/generate_question_variants.py \\
        --csv us_policy_event_battery_v4.csv \\
        --out-dir data_artifacts/question_variants \\
        --n-variants 100

    # Single-policy dry run:
    python scripts/data_prep/generate_question_variants.py \\
        --filter US-1935-SOCIAL-SECURITY \\
        --n-variants 100

Cost: ~$0.005 per OpenAI call. 211 policies × 5 calls/policy = ~$5 total
at default 100 variants.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PARAPHRASE_SYSTEM = """You are an expert at writing yes/no policy questions in varied styles.
Given a single policy question, produce N paraphrased variants that preserve the EXACT
meaning. Vary:
  - Syntactic structure (active/passive voice, clause ordering)
  - Vocabulary (synonyms, register)
  - Framing (legal/political/economic vocabulary)
  - Formality level
  - Length (some short, some longer)

Critical constraints:
  - Every variant must be a YES/NO question (answerable with Yes or No).
  - Every variant must convey the same underlying policy proposition.
  - Don't add new substantive content; just rephrase.

Return JSON with a single key "variants" whose value is a list of N strings.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path,
                   default=Path("us_policy_event_battery_v4.csv"),
                   help="Path to us_policy_event_battery_v4.csv.")
    p.add_argument("--out-dir", type=Path,
                   default=Path("data_artifacts/question_variants"),
                   help="Output dir; one <event_id>.jsonl file per policy.")
    p.add_argument("--n-variants", type=int, default=100,
                   help="Number of variants per policy (default 100).")
    p.add_argument("--batch-size", type=int, default=20,
                   help="Variants per OpenAI call (smaller = fewer max-token issues).")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--max-workers", type=int, default=16,
                   help="Concurrent OpenAI calls.")
    p.add_argument("--filter", nargs="+", default=None,
                   help="Only process these event_ids (smoke / dry run).")
    p.add_argument("--force", action="store_true",
                   help="Re-generate variants even if output file exists.")
    return p.parse_args()


def load_policies(csv_path: Path) -> list[dict]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    out: list[dict] = []
    for _, row in df.iterrows():
        question = str(row.get("benchmark_question", "")).strip()
        if not question or question.lower() == "nan":
            # Skip rows without a benchmark_question
            continue
        out.append({
            "id": str(row["event_id"]),
            "benchmark_question": question,
            "event_year": int(row["event_year"]),
        })
    return out


def generate_one_batch(client, model: str, question: str, n: int) -> list[str]:
    """Ask the model for n paraphrases of `question`. Returns a list."""
    user = (
        f"Generate {n} paraphrased variants of the following yes/no question.\n"
        f"\n"
        f"Question: {question}\n"
        f"\n"
        f"Return JSON {{\"variants\": [...]}}."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PARAPHRASE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.8,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    obj = json.loads(content)
    variants = obj.get("variants", [])
    return [v.strip() for v in variants if isinstance(v, str) and v.strip()]


def process_policy(client, model: str, policy: dict,
                   n_variants: int, batch_size: int,
                   out_dir: Path, force: bool) -> tuple[str, int, str | None]:
    event_id = policy["id"]
    out_path = out_dir / f"{event_id}.jsonl"
    if out_path.exists() and not force:
        n_existing = sum(1 for _ in open(out_path, encoding="utf-8"))
        return (event_id, n_existing, "exists")

    n_batches = (n_variants + batch_size - 1) // batch_size
    all_variants: list[str] = []
    for batch_idx in range(n_batches):
        n_this = min(batch_size, n_variants - len(all_variants))
        try:
            variants = generate_one_batch(client, model, policy["benchmark_question"], n_this)
        except Exception as e:  # noqa: BLE001
            return (event_id, len(all_variants), f"error_batch_{batch_idx}: {e}")
        all_variants.extend(variants)
        if len(all_variants) >= n_variants:
            break

    all_variants = all_variants[:n_variants]
    if len(all_variants) < n_variants * 0.5:
        return (event_id, len(all_variants), "too_few")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, v in enumerate(all_variants):
            f.write(json.dumps({
                "event_id": event_id,
                "variant_idx": idx,
                "variant_question": v,
            }, ensure_ascii=False) + "\n")
    return (event_id, len(all_variants), None)


def main() -> None:
    args = parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("OPENAI_API_KEY env var not set")
    from openai import OpenAI
    client = OpenAI()

    policies = load_policies(args.csv)
    if args.filter:
        keep = set(args.filter)
        policies = [p for p in policies if p["id"] in keep]
    if not policies:
        sys.exit("no policies to process (check --filter)")

    print(f"Processing {len(policies)} policies, {args.n_variants} variants each "
          f"(batch_size={args.batch_size}, workers={args.max_workers})")
    print(f"Output: {args.out_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **_):
            return it

    t0 = time.time()
    n_ok = n_skip = n_err = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(process_policy, client, args.model, p,
                             args.n_variants, args.batch_size,
                             args.out_dir, args.force): p["id"]
                   for p in policies}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="policies"):
            event_id, n, status = fut.result()
            if status == "exists":
                n_skip += 1
            elif status is None:
                n_ok += 1
            else:
                n_err += 1
                print(f"  ERR {event_id}: {status}")

    elapsed = time.time() - t0
    print()
    print(f"  done in {elapsed/60:.1f} min")
    print(f"  ok:       {n_ok}")
    print(f"  skipped:  {n_skip} (already existed; use --force to regenerate)")
    print(f"  errors:   {n_err}")


if __name__ == "__main__":
    main()
