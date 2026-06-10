"""Loader for the GST (Gentzkow-Shapiro-Taddy) Congressional speech corpus.

Hein 'bound' and 'daily' editions ship per-Congress files:
  descr_NNN.txt    pipe-delimited index — speech_id|chamber|date|...|word_count
  speeches_NNN.txt pipe-delimited text  — speech_id|speech
joined on speech_id; the calendar year comes from the descr `date` (YYYYMMDD),
and the descr `word_count` (last column) is authoritative — we never re-tokenize.

Coverage: bound Congresses 043-111 (~1873-2010), daily 097-114 (~1981-2016). We
read bound for years <=2010 and daily for years >=2011 so the 1981-2010 overlap
is not double-counted. Most speeches are short procedural remarks (word_count in
the single digits), so the default min_word_count=200 keeps only substantive
floor speeches — empirically ~7% of speeches in the 1930s rising to ~29% by the
2010s, still tens of thousands per year.

Filter keys:
    year: int | list[int]   restrict to one or more calendar years
    min_word_count: int     default 200
"""
from __future__ import annotations

from pathlib import Path

GST_ROOT = Path("C:/Users/danielyoon/Dropbox/SWM/raw/GST_speeches")
BOUND = GST_ROOT / "hein-bound"
DAILY = GST_ROOT / "hein-daily"

# descr columns, in order (pipe-delimited). We only need date and word_count.
_DATE_COL = 2
_WC_COL = 13
_N_COLS = 14

# Below this calendar year we read the bound edition; at/above it, the daily
# edition — the split that avoids the 1981-2010 bound/daily overlap.
_DAILY_FROM_YEAR = 2011


def _congress_of(path: Path) -> int:
    """Congress number embedded in a descr_/speeches_ filename."""
    return int(path.stem.split("_")[1])


def _candidate_congresses(year_lo: int, year_hi: int) -> range:
    """Congress numbers whose ~2-year span can contain [year_lo, year_hi].

    Congress C spans roughly [1789 + 2(C-1), +1]; a calendar year can also bleed
    into the prior Congress's lame-duck session, so we pad by one on each side.
    """
    clo = (year_lo - 1789) // 2 + 1 - 1
    chi = (year_hi - 1789) // 2 + 1 + 1
    return range(clo, chi + 1)


def _parse_descr(path: Path, min_wc: int, year_lo: int, year_hi: int) -> dict[str, tuple[int, int]]:
    """{speech_id: (year, word_count)} for speeches passing the year+length filters."""
    keep: dict[str, tuple[int, int]] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < _N_COLS:
                continue
            date = parts[_DATE_COL]
            if len(date) < 4 or not date[:4].isdigit():
                continue
            year = int(date[:4])
            if year < year_lo or year > year_hi:
                continue
            try:
                wc = int(parts[_WC_COL])
            except ValueError:
                continue
            if wc < min_wc:
                continue
            keep[parts[0]] = (year, wc)
    return keep


def _iter_congress(descr_path: Path, speeches_path: Path,
                   min_wc: int, year_lo: int, year_hi: int):
    """Yield passage dicts for one Congress, streaming the (large) speeches file."""
    keep = _parse_descr(descr_path, min_wc, year_lo, year_hi)
    if not keep:
        return
    cong = _congress_of(descr_path)
    with open(speeches_path, encoding="utf-8", errors="replace") as f:
        f.readline()  # header "speech_id|speech"
        for line in f:
            sid, sep, text = line.rstrip("\n").partition("|")
            if not sep:
                continue
            meta = keep.get(sid)
            if meta is None:
                continue
            text = text.strip()
            if not text:
                continue
            year, wc = meta
            yield {
                "doc_id": f"gst:{cong:03d}:{sid}",
                "source": "gst",
                "text": text,
                "year": year,
                "word_count": wc,
            }


def _edition_spans(year_lo: int, year_hi: int):
    """(root, ylo, yhi) segments splitting the request across bound/daily editions."""
    spans = []
    bound_hi = min(year_hi, _DAILY_FROM_YEAR - 1)
    if year_lo <= bound_hi:
        spans.append((BOUND, year_lo, bound_hi))
    daily_lo = max(year_lo, _DAILY_FROM_YEAR)
    if daily_lo <= year_hi:
        spans.append((DAILY, daily_lo, year_hi))
    return spans


def select(filter: dict, n: int | None = None) -> list[dict]:
    """Return up to n GST speech passages matching `filter`."""
    years = filter.get("year")
    if isinstance(years, int):
        years = [years]
    min_wc = filter.get("min_word_count", 200)
    year_lo = min(years) if years else 1789
    year_hi = max(years) if years else 9999
    year_set = set(years) if years else None

    out: list[dict] = []
    for root, ylo, yhi in _edition_spans(year_lo, year_hi):
        if not root.exists():
            continue
        for c in _candidate_congresses(ylo, yhi):
            descr = root / f"descr_{c:03d}.txt"
            speeches = root / f"speeches_{c:03d}.txt"
            if not (descr.exists() and speeches.exists()):
                continue
            for rec in _iter_congress(descr, speeches, min_wc, ylo, yhi):
                if year_set is not None and rec["year"] not in year_set:
                    continue
                out.append(rec)
                if n is not None and len(out) >= n:
                    return out
    return out
