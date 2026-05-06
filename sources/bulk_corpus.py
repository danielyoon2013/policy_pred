"""Loader for the bulk per-year corpus at D:/hist_LLM/corpus/raw/{year}/.

Each year directory holds 100 `subset_*.parquet` files. Documents have a
`collection` column (USPTO, US-PD-Newspapers, Caselaw Access Project,
Open-Science-Pile, OpenAlex, English-PD, etc.) — see CORPUS_SUMMARY.md
in the original hist_llm repo for full inventory.

Filter keys (all optional):
    collection: list[str]   keep docs whose `collection` field is in this list
    year: int | list[int]   restrict to one or more years (default: all years)
    language: list[str]     default ["English"]
    min_word_count: int     default 100
    max_word_count: int     no default (uncapped)
"""
from __future__ import annotations

from pathlib import Path

CORPUS_ROOT = Path("D:/hist_LLM/corpus/raw")


def select(filter: dict, n: int | None = None) -> list[dict]:
    """Return up to n docs matching `filter`.

    Each returned dict has: doc_id, source, text (and a few helpful extras
    so generators don't have to re-derive: collection, year, word_count).
    """
    import pandas as pd

    # Resolve year range.
    years = filter.get("year")
    if years is None:
        year_dirs = sorted(p for p in CORPUS_ROOT.iterdir() if p.is_dir() and p.name.isdigit())
    else:
        if isinstance(years, int):
            years = [years]
        year_dirs = [CORPUS_ROOT / str(y) for y in years]

    keep_collections = set(filter.get("collection", [])) or None
    keep_languages = set(filter.get("language", ["English"]))
    min_wc = filter.get("min_word_count", 100)
    max_wc = filter.get("max_word_count")

    out: list[dict] = []
    for year_dir in year_dirs:
        if not year_dir.exists():
            continue
        for parquet_file in sorted(year_dir.glob("subset_*.parquet")):
            df = pd.read_parquet(parquet_file)
            if keep_collections is not None:
                df = df[df["collection"].isin(keep_collections)]
            if keep_languages:
                df = df[df["language"].isin(keep_languages)]
            df = df[df["word_count"] >= min_wc]
            if max_wc is not None:
                df = df[df["word_count"] <= max_wc]

            for _, row in df.iterrows():
                text = row.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                out.append({
                    "doc_id": row["identifier"],
                    "source": f"bulk_corpus:{row['collection']}",
                    "text": text,
                    "collection": row["collection"],
                    "year": int(row["year"]),
                    "word_count": int(row["word_count"]),
                })
                if n is not None and len(out) >= n:
                    return out
    return out
