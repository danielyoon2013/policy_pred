"""Loader for The Economist weekly article archive.

D:/hist_LLM/additional_data/raw/news_archives/Economist/economist_YYYY-MMDD.parquet
— one weekly issue per file, one row per article. The calendar year is in the
filename. Columns include `ocr_text` (article text), `word_count` (stored as a
STRING — coerced here), `article_id`, `article_type`.

The pre-1954 issues are thin and the archive ends in 2014, so for this corpus we
default to min_word_count=100 (vs 200 for cleaner sources) to keep enough usable
articles; an OCR-garbage gate drops rows that are >50% non-alphanumeric (1962 is a
known OCR-corrupt year where ~66% of "articles" are mis-parsed ads/tables).

Filter keys:
    year: int | list[int]   restrict to one or more calendar years
    min_word_count: int     default 100
"""
from __future__ import annotations

from pathlib import Path

ECON_ROOT = Path("D:/hist_LLM/additional_data/raw/news_archives/Economist")


def _is_garbage(text: str) -> bool:
    """True if the text is mostly non-alphanumeric (OCR failure / ad tables)."""
    if not text:
        return True
    clean = sum(c.isalnum() or c.isspace() for c in text)
    return (clean / len(text)) < 0.5


def _to_int(val) -> int | None:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def select(filter: dict, n: int | None = None) -> list[dict]:
    """Return up to n Economist article passages matching `filter`."""
    import pandas as pd

    years = filter.get("year")
    if isinstance(years, int):
        years = [years]
    min_wc = filter.get("min_word_count", 100)

    if years:
        files = []
        for y in years:
            files += sorted(ECON_ROOT.glob(f"economist_{y}-*.parquet"))
    else:
        files = sorted(ECON_ROOT.glob("economist_*.parquet"))

    out: list[dict] = []
    for f in files:
        fy = _to_int(f.stem.split("_")[1][:4])
        if fy is None:
            continue
        df = pd.read_parquet(f)
        if "ocr_text" not in df.columns:
            continue
        has_wc = "word_count" in df.columns
        has_id = "article_id" in df.columns
        for i, row in df.iterrows():
            text = row.get("ocr_text")
            if not isinstance(text, str) or not text.strip():
                continue
            wc = _to_int(row.get("word_count")) if has_wc else len(text.split())
            if wc is None:
                wc = len(text.split())
            if wc < min_wc:
                continue
            text = text.strip()
            if _is_garbage(text):
                continue
            doc_id = str(row.get("article_id")) if has_id else f"{f.stem}:{i}"
            out.append({
                "doc_id": f"economist:{doc_id}",
                "source": "economist",
                "text": text,
                "year": fy,
                "word_count": wc,
            })
            if n is not None and len(out) >= n:
                return out
    return out
