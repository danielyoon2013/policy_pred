"""V2 stage 1: build W (seed docs) for one experiment.

Reads experiment YAML, dispatches to collection loaders by name, writes
combined output to D:/.../experiments/<name>/corpus.parquet.

Run:
    python corpus.py --experiment experiments/math_lora_r32.yaml

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

from policy_pred import config, sources  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True,
                   help="Path to experiment YAML (e.g. experiments/math_lora_r32.yaml).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing corpus.parquet for this experiment.")
    return p.parse_args()


def load_experiment(path: Path) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def experiment_dir(name: str) -> Path:
    """Resolve the experiment output dir under DATA_ROOT."""
    return config.DATA_ROOT / "experiments" / name


def main() -> None:
    args = parse_args()
    cfg = load_experiment(args.experiment)
    name = cfg["name"]
    out_dir = experiment_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "corpus.parquet"

    if out_path.exists() and not args.force:
        print(f"corpus.parquet already exists at {out_path} (use --force to overwrite)")
        return

    print(f"Building corpus for experiment '{name}'...")
    docs: list[dict] = []
    for col_cfg in cfg["corpus"]["collections"]:
        source = col_cfg["source"]
        filter_ = col_cfg.get("filter", {})
        n_each = col_cfg.get("n", None)
        loader = sources.get(source)
        print(f"  source={source}  filter={filter_}  n={n_each}")
        t0 = time.time()
        sub = loader.select(filter_, n=n_each)
        print(f"    -> {len(sub):,} docs ({time.time()-t0:.1f}s)")
        docs.extend(sub)

    n_seeds = cfg["corpus"].get("n_seeds")
    if n_seeds is not None and len(docs) > n_seeds:
        import random
        random.seed(cfg.get("seed", 42))
        random.shuffle(docs)
        docs = docs[:n_seeds]
        print(f"  truncated to n_seeds={n_seeds}")

    if not docs:
        sys.exit("no documents matched any collection filter; check the experiment YAML.")

    # Write parquet via pandas.
    import pandas as pd
    df = pd.DataFrame(docs)
    df.to_parquet(out_path, index=False)
    print(f"\nWrote {len(df):,} docs to {out_path}")

    # Snapshot the experiment config + a summary stub for later inspection.
    log_dir = out_dir / "log"
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / "corpus.json", "w", encoding="utf-8") as f:
        json.dump({
            "experiment": name,
            "n_docs": len(df),
            "by_source": df["source"].value_counts().to_dict(),
        }, f, indent=2)


if __name__ == "__main__":
    main()
