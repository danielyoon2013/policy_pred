"""Loader for the per-year NYT archives.

Reads D:/hist_LLM/additional_data/raw/news_archives/NYT_filtered_500char/nyt_{year}.parquet.
Each row is one article with `combined_text` (headline + abstract + lead),
`pub_date`, `word_count`, `headline_main`, etc. NYT data is summaries +
lead paragraphs, not full article bodies — typical doc is ~100-200 tokens.

Filter keys (all optional):
    year: int | list[int]   restrict to one or more years (default: all years)
    min_word_count: int     default 100
    max_word_count: int     no default (uncapped)
"""
from __future__ import annotations

from pathlib import Path

NYT_ROOT = Path("D:/hist_LLM/additional_data/raw/news_archives/NYT_filtered_500char")


def select(filter: dict, n: int | None = None) -> list[dict]:
    """Return up to n NYT articles matching `filter`."""
    import pandas as pd

    years = filter.get("year")
    if years is None:
        files = sorted(NYT_ROOT.glob("nyt_*.parquet"))
    else:
        if isinstance(years, int):
            years = [years]
        files = [NYT_ROOT / f"nyt_{y}.parquet" for y in years]

    min_wc = filter.get("min_word_count", 100)
    max_wc = filter.get("max_word_count")

    out: list[dict] = []
    for f in files:
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        df = df[df["word_count"] >= min_wc]
        if max_wc is not None:
            df = df[df["word_count"] <= max_wc]
        for _, row in df.iterrows():
            text = row.get("combined_text")
            if not isinstance(text, str) or not text.strip():
                continue
            year = row.get("file_year") or row.get("pub_date")
            out.append({
                "doc_id": str(row["_id"]),
                "source": f"nyt:{year}",
                "text": text,
                "year": int(year) if isinstance(year, (int, float)) else None,
                "headline": row.get("headline_main", ""),
                "word_count": int(row["word_count"]),
            })
            if n is not None and len(out) >= n:
                return out
    return out
