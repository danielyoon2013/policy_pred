"""Ingest the SWM discourse sources into per-year passage parquets.

Writes  D:/hist_LLM/policy_pred/swm_seeds/<source>/<Y>.parquet  with columns
[doc_id, source, text, year, word_count] — the raw passage pool each per-year
discourse block is later sampled from (sample_swm_blocks.py).

GST is organized by Congress, so it is ingested in a single streaming pass per
Congress (bucketing speeches by their calendar year, flushing finished years to
bound memory). Economist and FOMC are per-year, so they go through
source.select(year=Y) directly. FOMC PDF extraction is slow, so its per-year
fetch is capped (--fomc-cap) — the block only needs a few thousand passages/yr.

Usage:
  python scripts/data_prep/ingest_swm.py --source gst       --start-year 1932 --end-year 2016
  python scripts/data_prep/ingest_swm.py --source economist --start-year 1931 --end-year 2014
  python scripts/data_prep/ingest_swm.py --source fomc      --start-year 1936 --end-year 2019
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_ROOT = Path("D:/hist_LLM/policy_pred/swm_seeds")

# Default per-source length filters (per the supply census + plan).
DEFAULT_MIN_WC = {"gst": 200, "economist": 100, "fomc": 200}


def _congress_start_year(c: int) -> int:
    return 1789 + 2 * (c - 1)


def _write_year(out_dir: Path, year: int, recs: list[dict], overwrite: bool) -> bool:
    import pandas as pd
    path = out_dir / f"{year}.parquet"
    if path.exists() and not overwrite:
        print(f"  {year}: SKIP (exists, {path.name})")
        return False
    if not recs:
        print(f"  {year}: 0 passages — not written")
        return False
    pd.DataFrame(recs, columns=["doc_id", "source", "text", "year", "word_count"]
                 ).to_parquet(path, index=False)
    print(f"  {year}: wrote {len(recs)} passages -> {path.name}")
    return True


def ingest_gst(start: int, end: int, out_dir: Path, min_wc: int, overwrite: bool) -> None:
    from sources import gst
    buckets: dict[int, list[dict]] = {}

    def flush(below_year: int) -> None:
        for y in sorted(k for k in buckets if k < below_year):
            _write_year(out_dir, y, buckets.pop(y), overwrite)

    for root, ylo, yhi in gst._edition_spans(start, end):
        if not root.exists():
            print(f"  (edition root missing: {root})")
            continue
        for c in gst._candidate_congresses(ylo, yhi):
            descr = root / f"descr_{c:03d}.txt"
            speeches = root / f"speeches_{c:03d}.txt"
            if not (descr.exists() and speeches.exists()):
                continue
            print(f"  Congress {c:03d} ({root.name}) ...")
            for rec in gst._iter_congress(descr, speeches, min_wc, ylo, yhi):
                y = rec["year"]
                if start <= y <= end:
                    buckets.setdefault(y, []).append(rec)
            flush(_congress_start_year(c))   # finished years can be written now
    for y in sorted(buckets):                # flush the tail
        _write_year(out_dir, y, buckets.pop(y), overwrite)


def ingest_per_year(source: str, start: int, end: int, out_dir: Path,
                    min_wc: int, overwrite: bool, cap: int | None) -> None:
    import importlib
    mod = importlib.import_module(f"sources.{source}")
    for year in range(start, end + 1):
        path = out_dir / f"{year}.parquet"
        if path.exists() and not overwrite:
            print(f"  {year}: SKIP (exists)")
            continue
        flt = {"year": year, "min_word_count": min_wc}
        recs = mod.select(flt, n=cap)
        _write_year(out_dir, year, recs, overwrite)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, choices=["gst", "economist", "fomc"])
    p.add_argument("--start-year", type=int, required=True)
    p.add_argument("--end-year", type=int, required=True)
    p.add_argument("--min-word-count", type=int, default=None,
                   help="override the per-source default length filter.")
    p.add_argument("--fomc-cap", type=int, default=2500,
                   help="max passages/year for FOMC (PDF extraction is slow).")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    min_wc = args.min_word_count if args.min_word_count is not None \
        else DEFAULT_MIN_WC[args.source]
    out_dir = OUT_ROOT / args.source
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ingest {args.source}: {args.start_year}..{args.end_year}  "
          f"min_word_count={min_wc}  -> {out_dir}")
    if args.source == "gst":
        ingest_gst(args.start_year, args.end_year, out_dir, min_wc, args.overwrite)
    else:
        cap = args.fomc_cap if args.source == "fomc" else None
        ingest_per_year(args.source, args.start_year, args.end_year, out_dir,
                        min_wc, args.overwrite, cap)
    print("DONE")


if __name__ == "__main__":
    main()
