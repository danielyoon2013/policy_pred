"""V2 stage 2 (f): generate S from W via configured generators.

Reads experiment YAML + corpus.parquet, dispatches to each generator named
in `synth.generators`, writes combined chat-format JSONL to
D:/.../experiments/<name>/synth.jsonl.

Run:
    python synth.py --experiment experiments/math_lora_r32.yaml

Requires that `corpus.py --experiment ...` already produced the input.
Idempotent: skip if output exists, --force to overwrite.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make policy_pred.* importable when run as a script from the repo root.
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from policy_pred import config, generators  # noqa: E402
from policy_pred.corpus import experiment_dir, load_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing synth.jsonl for this experiment.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of seed docs (for quick smoke runs).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment(args.experiment)
    name = cfg["name"]
    out_dir = experiment_dir(name)
    out_path = out_dir / "synth.jsonl"

    if out_path.exists() and not args.force:
        print(f"synth.jsonl already exists at {out_path} (use --force to overwrite)")
        return

    corpus_path = out_dir / "corpus.parquet"
    if not corpus_path.exists():
        sys.exit(f"missing {corpus_path}; run corpus.py --experiment first.")

    import pandas as pd
    seeds = pd.read_parquet(corpus_path).to_dict(orient="records")
    if args.limit:
        seeds = seeds[: args.limit]
    print(f"Loaded {len(seeds):,} seed docs from {corpus_path}")

    out_records: list[dict] = []
    for gen_name, gen_cfg in cfg["synth"]["generators"].items():
        # Inherit top-level openai cfg if generator doesn't override.
        merged = {**cfg["synth"].get("openai", {}), **gen_cfg}
        Generator = generators.get(gen_name)
        gen = Generator(merged)
        print(f"\nRunning generator '{gen_name}' on {len(seeds):,} seeds...")
        t0 = time.time()
        sub = gen.generate(seeds)
        print(f"  -> {len(sub):,} records ({time.time()-t0:.1f}s)")
        out_records.extend(sub)

    if not out_records:
        sys.exit("no records produced by any generator; check generator configs.")

    # Write JSONL.
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(out_records):,} records to {out_path}")

    log_dir = out_dir / "log"
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / "synth.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": name,
            "n_seeds": len(seeds),
            "n_records": len(out_records),
            "generators": list(cfg["synth"]["generators"].keys()),
        }, f, indent=2)


if __name__ == "__main__":
    main()
