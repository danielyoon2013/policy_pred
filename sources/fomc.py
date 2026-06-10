"""Loader for the scraped FOMC (Federal Open Market Committee) documents.

C:/Users/danielyoon/Dropbox/SWM/raw/scrape_FOMC/<Y>/<Y>-MM-DD/
  minutes.pdf / transcript.pdf / tealbook*.pdf / statement.html / minutes.html ...

We extract text from the meeting **minutes**, **transcripts**, and **statements**
(skipping tealbooks, which are mostly forecast tables, not deliberation), then chunk
each document into ~paragraph-sized passages so a long transcript yields many usable
LoRA seeds. The calendar year comes from the year directory name.

Coverage ~1936-2019. Early meetings are minutes-only and short (a handful of passages
per year — a garnish); post-1976 transcripts are long, so FOMC contributes more
monetary-policy discourse in the modern era.

Filter keys:
    year: int | list[int]   restrict to one or more calendar years
    min_word_count: int     default 200 (per-chunk floor)
    chunk_words: int        target words per emitted passage (default 400)
"""
from __future__ import annotations

import re
from pathlib import Path

FOMC_ROOT = Path("C:/Users/danielyoon/Dropbox/SWM/raw/scrape_FOMC")

# Which document types to read (glob patterns within a meeting dir). Tealbooks are
# excluded on purpose (tabular forecasts, not discourse).
_DOC_GLOBS = ("minutes*.pdf", "transcript*.pdf", "minutes*.html", "*statement*.html")

_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _read_pdf(path: Path) -> str:
    try:
        import pdfplumber
    except ImportError:
        return ""
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    except Exception:
        return ""


def _read_html(path: Path) -> str:
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return _HTML_TAG.sub(" ", raw)


def _chunk(text: str, chunk_words: int, min_wc: int):
    """Split text into ~chunk_words passages, dropping chunks below min_wc."""
    words = _WS.sub(" ", text).strip().split(" ")
    if not words or words == [""]:
        return
    for i in range(0, len(words), chunk_words):
        piece = words[i:i + chunk_words]
        if len(piece) < min_wc:
            continue
        yield " ".join(piece), len(piece)


def _year_dirs(years):
    if years is None:
        return sorted(p for p in FOMC_ROOT.iterdir()
                      if p.is_dir() and p.name.isdigit())
    return [FOMC_ROOT / str(y) for y in years]


def select(filter: dict, n: int | None = None) -> list[dict]:
    """Return up to n FOMC document passages matching `filter`."""
    years = filter.get("year")
    if isinstance(years, int):
        years = [years]
    min_wc = filter.get("min_word_count", 200)
    chunk_words = filter.get("chunk_words", 400)

    out: list[dict] = []
    for ydir in _year_dirs(years):
        if not ydir.exists():
            continue
        year = int(ydir.name)
        for meeting in sorted(p for p in ydir.iterdir() if p.is_dir()):
            docs = []
            for pat in _DOC_GLOBS:
                docs += sorted(meeting.glob(pat))
            for doc in docs:
                text = _read_pdf(doc) if doc.suffix.lower() == ".pdf" else _read_html(doc)
                if not text.strip():
                    continue
                for j, (passage, wc) in enumerate(_chunk(text, chunk_words, min_wc)):
                    out.append({
                        "doc_id": f"fomc:{meeting.name}:{doc.stem}:{j}",
                        "source": "fomc",
                        "text": passage,
                        "year": year,
                        "word_count": wc,
                    })
                    if n is not None and len(out) >= n:
                        return out
    return out
