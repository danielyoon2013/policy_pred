"""Inspect the historical corpus to understand what 'collections' exist per year.

Walks the Tier 1 (newswire, NYT, FT) and Tier 2 (corpus/raw/{year}/subset_*.parquet)
sources for years 1930-1940 and reports:
  - schema (column names + dtypes) per source
  - row counts
  - distribution of any 'source' / 'publication' / 'collection' style field
  - 2-3 sample documents (truncated)

Output is plain text to stdout; nothing written to disk. Run from repo root:
    python scripts/inspect_corpus.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CORPUS_RAW = Path("D:/hist_LLM/corpus/raw")
ADDL = Path("D:/hist_LLM/additional_data/raw")
NEWSWIRE = ADDL / "newswire"
NYT = ADDL / "news_archives/NYT_filtered_500char"
FT = ADDL / "news_archives/FT"

YEARS = list(range(1930, 1941))  # 1930..1940 inclusive


def head(s: str, n: int = 200) -> str:
    s = " ".join(s.split())
    return s[:n] + ("..." if len(s) > n else "")


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def inspect_one_subset_schema(year: int) -> None:
    year_dir = CORPUS_RAW / str(year)
    if not year_dir.exists():
        print(f"  [missing] {year_dir}")
        return
    parquets = sorted(year_dir.glob("subset_*.parquet"))
    if not parquets:
        print(f"  [no parquets] {year_dir}")
        return
    df = pd.read_parquet(parquets[0])
    print(f"  {year_dir} ({len(parquets)} subset files)")
    print(f"    schema of {parquets[0].name}:")
    for col, dtype in zip(df.columns, df.dtypes):
        print(f"      {col:30s}  {dtype}")
    print(f"    rows in this one file: {len(df):,}")
    # Sample
    if len(df) > 0:
        sample = df.iloc[0]
        print(f"    first row keys/values (truncated):")
        for col in df.columns:
            val = sample[col]
            if isinstance(val, str):
                val = head(val, 150)
            print(f"      {col:30s}  {val}")


def aggregate_subset_metadata(year: int, max_files: int = 100) -> None:
    """Look across all subset files for a year and report value distributions
    on any column that smells like a collection/source/publication marker."""
    year_dir = CORPUS_RAW / str(year)
    if not year_dir.exists():
        return
    parquets = sorted(year_dir.glob("subset_*.parquet"))[:max_files]
    if not parquets:
        return

    METADATA_COLS = [
        "collection", "source", "publication", "publisher",
        "newspaper", "title", "type", "category", "doc_type",
    ]

    counts: dict[str, dict] = {}
    total_rows = 0
    for p in parquets:
        df = pd.read_parquet(p, columns=None)
        total_rows += len(df)
        for col in METADATA_COLS:
            if col in df.columns:
                vc = df[col].value_counts(dropna=False)
                if col not in counts:
                    counts[col] = {}
                for k, v in vc.items():
                    counts[col][k] = counts[col].get(k, 0) + int(v)

    print(f"  {year}: {len(parquets)} files, {total_rows:,} rows total")
    if not counts:
        print(f"    (no metadata columns matched: {METADATA_COLS})")
        return
    for col, dist in counts.items():
        items = sorted(dist.items(), key=lambda kv: -kv[1])
        print(f"    {col} ({len(items)} unique):")
        for k, v in items[:20]:
            kstr = str(k)[:40]
            print(f"      {v:>10,}  {kstr}")
        if len(items) > 20:
            print(f"      ... +{len(items) - 20} more")


def inspect_newswire(year: int) -> None:
    p = NEWSWIRE / f"{year}_data_clean.json"
    if not p.exists():
        print(f"  [missing] {p}")
        return
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        print(f"  {p.name}: list of {len(data):,} items")
        if data:
            sample = data[0]
            if isinstance(sample, dict):
                print(f"    keys: {list(sample.keys())}")
                for k, v in sample.items():
                    if isinstance(v, str):
                        v = head(v, 150)
                    print(f"      {k:25s}  {v}")
    elif isinstance(data, dict):
        print(f"  {p.name}: dict with keys {list(data.keys())[:10]}")


def inspect_parquet(path: Path) -> None:
    if not path.exists():
        print(f"  [missing] {path}")
        return
    df = pd.read_parquet(path)
    print(f"  {path.name}: {len(df):,} rows, columns: {list(df.columns)}")
    if len(df) > 0:
        sample = df.iloc[0]
        for col in df.columns:
            val = sample[col]
            if isinstance(val, str):
                val = head(val, 150)
            print(f"    {col:25s}  {val}")


def main() -> None:
    section("TIER 1 — Newswire (one JSON per year)")
    for y in [1931, 1935, 1940]:
        inspect_newswire(y)

    section("TIER 1 — NYT (one parquet per year)")
    for y in [1931, 1935, 1940]:
        inspect_parquet(NYT / f"nyt_{y}.parquet")

    section("TIER 1 — FT (one parquet per year)")
    for y in [1931, 1935, 1940]:
        inspect_parquet(FT / f"{y}.parquet")

    section("TIER 2 — bulk corpus subset schema (year 1931, file 0)")
    inspect_one_subset_schema(1931)

    section("TIER 2 — bulk corpus collection/source breakdown by year (1930-1940)")
    for y in YEARS:
        aggregate_subset_metadata(y)


if __name__ == "__main__":
    main()
