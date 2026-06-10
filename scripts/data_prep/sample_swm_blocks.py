"""Sample per-year discourse seed blocks for the SWM LoRA ablation.

Reads the ingested passage pools (D:/.../swm_seeds/<source>/<Y>.parquet) and writes
the per-year seed blocks the synth step consumes (one synth doc per seed, n_per_seed=1):

  C:/tmp/policy_pred/years/<Y>/seed_swmgst.parquet      -- 5000 GST passages          (Arm S1)
  C:/tmp/policy_pred/years/<Y>/seed_swmecofomc.parquet  -- 3000 Economist + FOMC       (Arm S2 extra)

The Econ+FOMC block takes ALL available FOMC (capped at the target), then fills the
remainder with Economist. Blocks are sampled with a per-year fixed RNG (reproducible).
Years short of target use what's available and are LOGGED (never faked / padded).

Usage:
  python scripts/data_prep/sample_swm_blocks.py --start-year 1932 --end-year 2016
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

SEEDS_ROOT = Path("D:/hist_LLM/policy_pred/swm_seeds")
YEARS_ROOT = Path("C:/tmp/policy_pred/years")
_COLS = ["doc_id", "source", "text", "year", "word_count"]


def _load(source: str, year: int):
    import pandas as pd
    p = SEEDS_ROOT / source / f"{year}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=_COLS)
    return pd.read_parquet(p)


def _sample(df, k: int, rng: random.Random):
    if k <= 0 or len(df) == 0:
        return df.iloc[:0]
    if len(df) <= k:
        return df
    return df.iloc[sorted(rng.sample(range(len(df)), k))]


def build_gst_block(year: int, target: int, overwrite: bool) -> tuple[int, bool]:
    out = YEARS_ROOT / str(year) / "seed_swmgst.parquet"
    if out.exists() and not overwrite:
        import pandas as pd
        return len(pd.read_parquet(out)), False
    df = _load("gst", year)
    block = _sample(df, target, random.Random(year))
    out.parent.mkdir(parents=True, exist_ok=True)
    block.to_parquet(out, index=False)
    return len(block), len(block) < target


def build_ecofomc_block(year: int, target: int, overwrite: bool) -> tuple[int, int, int, bool]:
    import pandas as pd
    out = YEARS_ROOT / str(year) / "seed_swmecofomc.parquet"
    if out.exists() and not overwrite:
        df = pd.read_parquet(out)
        nf = int((df["source"] == "fomc").sum()) if "source" in df else 0
        return len(df), nf, len(df) - nf, False
    rng = random.Random(year * 7 + 1)
    fomc = _sample(_load("fomc", year), target, rng)             # all FOMC, capped
    econ = _sample(_load("economist", year), target - len(fomc), rng)  # Econ fills rest
    block = pd.concat([fomc, econ], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    block.to_parquet(out, index=False)
    return len(block), len(fomc), len(econ), len(block) < target


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-year", type=int, required=True)
    p.add_argument("--end-year", type=int, required=True)
    p.add_argument("--target-gst", type=int, default=5000)
    p.add_argument("--target-ecofomc", type=int, default=3000)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    short_gst, short_eco = [], []
    print(f"{'year':>5} {'GST':>7} {'Eco+FOMC':>10} {'(fomc':>7} {'econ)':>7}")
    for year in range(args.start_year, args.end_year + 1):
        n_gst, sg = build_gst_block(year, args.target_gst, args.overwrite)
        n_ef, n_f, n_e, se = build_ecofomc_block(year, args.target_ecofomc, args.overwrite)
        flag = "  <- SHORT" if (sg or se) else ""
        print(f"{year:>5} {n_gst:>7} {n_ef:>10} {n_f:>7} {n_e:>7}{flag}")
        if sg:
            short_gst.append((year, n_gst))
        if se:
            short_eco.append((year, n_ef))

    print("\n=== shortfalls (used available, logged) ===")
    print(f"  GST < {args.target_gst}:      {short_gst or 'none'}")
    print(f"  Eco+FOMC < {args.target_ecofomc}: {short_eco or 'none'}")


if __name__ == "__main__":
    main()
