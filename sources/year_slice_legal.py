"""Slice the legal-text subset of the bulk corpus for one year.

Reads `D:/hist_LLM/corpus/raw/{year}/subset_*.parquet`, filters to
`collection == "Caselaw Access Project"`, returns/writes a single
parquet with normalized columns: doc_id, source, text, year, word_count.

Reuses the bulk_corpus loader's filter logic; the only specialization is
hard-coding the Caselaw collection and producing a single output parquet
suitable for use as seeds by `synth_naive/run.py` or `synth_2step/run.py`.

Run from the repo root:
    python sources/year_slice_legal.py --year 1931 \
        --out D:/hist_LLM/policy_pred/years/1931/legal.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make policy_pred.* importable when run as a script from the repo root.
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent.parent))

from policy_pred.sources import bulk_corpus  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, required=True,
                   help="Year to slice (e.g. 1931).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output parquet path. Parent dir created if missing.")
    p.add_argument("--collections", nargs="+",
                   default=["Caselaw Access Project"],
                   help="Bulk-corpus collection names to keep "
                        "(default: Caselaw Access Project).")
    p.add_argument("--min-word-count", type=int, default=200,
                   help="Drop docs with fewer words (default 200).")
    p.add_argument("--max-word-count", type=int, default=None,
                   help="Optional cap (default: no cap).")
    p.add_argument("--n", type=int, default=None,
                   help="Optional cap on total docs returned (default: all).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Slicing {args.year} legal corpus...")
    print(f"  collections: {args.collections}")
    print(f"  min_word_count: {args.min_word_count}")
    if args.max_word_count:
        print(f"  max_word_count: {args.max_word_count}")
    if args.n:
        print(f"  cap: {args.n}")

    docs = bulk_corpus.select(
        filter={
            "collection": args.collections,
            "year": args.year,
            "min_word_count": args.min_word_count,
            **({"max_word_count": args.max_word_count} if args.max_word_count else {}),
        },
        n=args.n,
    )

    if not docs:
        sys.exit(f"no docs found for year={args.year}, collections={args.collections}")

    # Write parquet with the columns synth_naive/run.py and synth_2step/run.py
    # know how to read (they look for `text`).
    import pandas as pd
    df = pd.DataFrame(docs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    # Summary.
    print(f"\nWrote {len(df):,} docs to {args.out}")
    print(f"  median word_count: {int(df['word_count'].median())}")
    print(f"  total tokens (approx, words×1.3): {int(df['word_count'].sum() * 1.3):,}")
    if "collection" in df.columns:
        print(f"  by collection:")
        for col, n in df["collection"].value_counts().items():
            print(f"    {col}: {n:,}")


if __name__ == "__main__":
    main()
