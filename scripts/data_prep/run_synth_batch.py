"""synth_naive via OpenAI Batch API: submit, poll, parse — all in one script.

The sync synth_naive flow makes one OpenAI call per seed serially. For a
75-year campaign at 5k seeds/year (375K calls) this takes ~12 hr at high
parallelism and costs ~$375. The batch API processes the same requests
asynchronously at half price (~$190) with the local machine idle.

This script wraps the full lifecycle:
  1. SUBMIT — load seeds, build per-year batch input JSONL, upload, create batch
  2. POLL   — wait until all batches complete
  3. PARSE  — download outputs, parse JSON responses, write per-year synth.jsonl
              matching the format produced by synth_naive/run.py exactly

Idempotent: state file (`batch_state.json` in --batch-work-dir) tracks each
year's batch id. Re-running the script picks up where it left off.

Cost: per-year ~$2.50 at batch rate (vs $5 sync). 75 years ≈ $190.
Wall time: typically 1-6 hr per batch (24h SLA worst case; batches are
independent so they complete in parallel server-side).

Usage:
    export OPENAI_API_KEY=sk-...
    # Submit + poll + parse for years 1946-2020:
    python scripts/data_prep/run_synth_batch.py --start-year 1946 --end-year 2020

    # Single-year smoke test (uses small seed count):
    python scripts/data_prep/run_synth_batch.py \\
        --start-year 1946 --end-year 1946 --n-seeds 100

    # If submit fails partway: re-run, it skips years already submitted.
    # If you just want to poll status:
    python scripts/data_prep/run_synth_batch.py \\
        --start-year 1946 --end-year 2020 --phase poll-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SYNTH_NAIVE_ROOT = REPO_ROOT / "synth" / "naive"
SYSTEM_PROMPT = (SYNTH_NAIVE_ROOT / "prompts" / "system.md").read_text(encoding="utf-8")
USER_TEMPLATE = (SYNTH_NAIVE_ROOT / "prompts" / "user.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-year", type=int, required=True)
    p.add_argument("--end-year", type=int, required=True)
    p.add_argument("--n-seeds", type=int, default=5000,
                   help="Cap seeds per year (default 5000).")
    p.add_argument("--n-per-seed", type=int, default=4,
                   help="Documents requested per seed (default 4).")
    p.add_argument("--max-seed-chars", type=int, default=4000,
                   help="Truncate seed text to N chars (default 4000).")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--corpus-root", type=Path,
                   default=Path("C:/tmp/policy_pred/years"),
                   help="Where per-year seed parquets live.")
    p.add_argument("--seed-parquet-name", default="legal.parquet",
                   help="Per-year seed parquet filename under corpus-root/<Y>/ "
                        "(default legal.parquet; e.g. seed_swmgst.parquet for the GST block).")
    p.add_argument("--out-subdir", default="naive",
                   help="Per-year output subdir for synth.jsonl "
                        "(default naive; e.g. naive_swmgst for the GST block).")
    p.add_argument("--batch-work-dir", type=Path,
                   default=Path("C:/tmp/policy_pred/batch_work"),
                   help="Where batch input/output JSONLs + state file live.")
    p.add_argument("--phase",
                   choices=["all", "submit-only", "poll-only", "parse-only"],
                   default="all",
                   help="Run a subset of phases (default: all three sequentially).")
    p.add_argument("--poll-interval", type=int, default=120,
                   help="Seconds between poll checks (default 120).")
    p.add_argument("--max-poll-hours", type=float, default=26.0,
                   help="Stop polling after N hours (default 26).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Seed loading (matches synth_naive's parquet detection)
# ---------------------------------------------------------------------------

def load_seeds(parquet_path: Path, limit: int, max_chars: int) -> list[str]:
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    for col in ("text_cleaned", "combined_text", "cleaned_article", "text", "article"):
        if col in df.columns:
            seeds = [t for t in df[col].dropna().tolist()
                     if isinstance(t, str) and t.strip()]
            return [s[:max_chars] for s in seeds[:limit]]
    raise KeyError(
        f"no text column found in {parquet_path}; "
        f"expected one of: text_cleaned, combined_text, cleaned_article, text, article"
    )


# ---------------------------------------------------------------------------
# Phase 1: build + submit per-year batches
# ---------------------------------------------------------------------------

def build_batch_input(year: int, seeds: list[str], args, out_path: Path) -> None:
    """Write OpenAI batch input JSONL for one year. One request per seed."""
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, seed in enumerate(seeds):
            user_content = USER_TEMPLATE.format(seed=seed, n_per_seed=args.n_per_seed)
            req = {
                "custom_id": f"year={year};seed={idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "response_format": {"type": "json_object"},
                },
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def phase_submit(client, args, state: dict, state_path: Path) -> None:
    print()
    print("=" * 60)
    print("  PHASE 1: SUBMIT")
    print("=" * 60)

    for year in range(args.start_year, args.end_year + 1):
        key = str(year)

        # Already submitted? Skip.
        if state.get(key, {}).get("batch_id"):
            print(f"  {year}: already submitted (batch_id={state[key]['batch_id']})")
            continue

        # Already have final synth.jsonl? Skip (sync run produced it).
        out_file = args.corpus_root / str(year) / args.out_subdir / "synth.jsonl"
        if out_file.exists():
            n = sum(1 for _ in open(out_file, encoding="utf-8"))
            print(f"  {year}: SKIP (existing synth.jsonl has {n} records)")
            continue

        parquet = args.corpus_root / str(year) / args.seed_parquet_name
        if not parquet.exists():
            print(f"  {year}: SKIP (no corpus parquet at {parquet})")
            continue

        print(f"  {year}: loading seeds from {parquet.name}...")
        seeds = load_seeds(parquet, args.n_seeds, args.max_seed_chars)
        if not seeds:
            print(f"  {year}: SKIP (no seeds)")
            continue

        input_path = args.batch_work_dir / f"input_{year}.jsonl"
        build_batch_input(year, seeds, args, input_path)
        print(f"  {year}: built batch input ({len(seeds)} requests, "
              f"{input_path.stat().st_size // 1024} KB)")

        with open(input_path, "rb") as f:
            file_obj = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"year": str(year)},
        )

        state[key] = {
            "batch_id": batch.id,
            "input_file_id": file_obj.id,
            "n_requests": len(seeds),
            "status": batch.status,
            "submitted_at": time.time(),
        }
        save_state(state_path, state)
        print(f"  {year}: submitted batch {batch.id} (status: {batch.status})")


# ---------------------------------------------------------------------------
# Phase 2: poll until all complete
# ---------------------------------------------------------------------------

def phase_poll(client, args, state: dict, state_path: Path) -> None:
    print()
    print("=" * 60)
    print("  PHASE 2: POLL")
    print("=" * 60)

    deadline = time.time() + args.max_poll_hours * 3600
    while True:
        active_keys = [k for k in state if state[k].get("status")
                       in ("validating", "in_progress", "finalizing", "submitted")]
        if not active_keys:
            print("  All batches in terminal status. Polling done.")
            return

        if time.time() > deadline:
            print(f"  Poll deadline ({args.max_poll_hours}h) reached. Stopping.")
            return

        print(f"\n  Checking status of {len(active_keys)} active batch(es)...")
        for key in active_keys:
            bid = state[key]["batch_id"]
            batch = client.batches.retrieve(bid)
            old_status = state[key].get("status")
            state[key]["status"] = batch.status
            if batch.status in ("completed", "failed", "expired", "cancelled"):
                state[key]["output_file_id"] = batch.output_file_id
                state[key]["error_file_id"] = batch.error_file_id
                state[key]["completed_at"] = time.time()
            if batch.status != old_status:
                rc = (batch.request_counts.completed if batch.request_counts else None)
                rt = (batch.request_counts.total if batch.request_counts else None)
                print(f"  year={key}: {old_status} -> {batch.status}"
                      f" ({rc}/{rt} requests)")
            else:
                rc = (batch.request_counts.completed if batch.request_counts else None)
                rt = (batch.request_counts.total if batch.request_counts else None)
                print(f"  year={key}: {batch.status} ({rc}/{rt})")
            save_state(state_path, state)

        # Recompute active list after status updates.
        still_active = [k for k in state if state[k].get("status")
                        in ("validating", "in_progress", "finalizing", "submitted")]
        if not still_active:
            print("  All batches reached terminal status.")
            return

        print(f"  {len(still_active)} still active; sleeping {args.poll_interval}s...")
        time.sleep(args.poll_interval)


# ---------------------------------------------------------------------------
# Phase 3: parse outputs, write per-year synth.jsonl
# ---------------------------------------------------------------------------

def parse_documents(content: str) -> list[str]:
    """Same parser as synth_naive/run.py: pulls 'documents' list from JSON."""
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


def phase_parse(client, args, state: dict, state_path: Path) -> None:
    print()
    print("=" * 60)
    print("  PHASE 3: PARSE + WRITE per-year synth.jsonl")
    print("=" * 60)

    for year_key in sorted(state.keys(), key=int):
        info = state[year_key]
        if info.get("status") != "completed":
            print(f"  year={year_key}: skip (status={info.get('status')})")
            continue
        if info.get("written"):
            print(f"  year={year_key}: skip (already written, {info.get('n_records')} records)")
            continue

        output_file_id = info.get("output_file_id")
        if not output_file_id:
            print(f"  year={year_key}: no output_file_id; skip")
            continue

        print(f"  year={year_key}: downloading output...")
        resp = client.files.content(output_file_id)
        output_text = resp.text

        out_dir = args.corpus_root / year_key / args.out_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "synth.jsonl"

        n_records = 0
        n_failures = 0
        with open(out_path, "w", encoding="utf-8") as outf:
            for line in output_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                custom_id = rec.get("custom_id", "")
                # custom_id format: year=1942;seed=487
                parts = {}
                for p in custom_id.split(";"):
                    if "=" in p:
                        k, v = p.split("=", 1)
                        parts[k] = v
                seed_idx = int(parts.get("seed", -1))

                if rec.get("error"):
                    n_failures += 1
                    continue
                body = rec.get("response", {}).get("body", {})
                choices = body.get("choices", [])
                if not choices:
                    n_failures += 1
                    continue
                content = choices[0]["message"]["content"]
                docs = parse_documents(content)
                if not docs:
                    n_failures += 1
                    continue
                for d in docs:
                    outf.write(json.dumps({
                        "text": d,
                        "metadata": {
                            "seed_idx": seed_idx,
                            "generator": "synth_naive_batch",
                            "year": int(year_key),
                        },
                    }, ensure_ascii=False) + "\n")
                    n_records += 1

        info["written"] = True
        info["n_records"] = n_records
        info["n_failures"] = n_failures
        save_state(state_path, state)
        print(f"  year={year_key}: wrote {n_records} records ({n_failures} failures) -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("OPENAI_API_KEY env var not set")
    from openai import OpenAI
    client = OpenAI()

    args.batch_work_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.batch_work_dir / "batch_state.json"
    state = load_state(state_path)

    print(f"Years to process: {args.start_year}..{args.end_year} "
          f"({args.end_year - args.start_year + 1} years)")
    print(f"State file:       {state_path}")
    print(f"Phases:           {args.phase}")

    if args.phase in ("all", "submit-only"):
        phase_submit(client, args, state, state_path)

    if args.phase in ("all", "poll-only"):
        phase_poll(client, args, state, state_path)

    if args.phase in ("all", "parse-only"):
        phase_parse(client, args, state, state_path)

    # Final summary.
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    n_total = n_completed = n_written = 0
    for key in sorted(state.keys(), key=int):
        info = state[key]
        n_total += 1
        if info.get("status") == "completed":
            n_completed += 1
        if info.get("written"):
            n_written += 1
    print(f"  Submitted: {n_total}")
    print(f"  Completed: {n_completed}")
    print(f"  Written:   {n_written}")


if __name__ == "__main__":
    main()
