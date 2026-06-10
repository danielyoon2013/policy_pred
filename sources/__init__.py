"""Plug-in registry for corpus source loaders.

A "source" is a way to pull seed documents from D: (NYT, FT, bulk corpus,
newswire, etc.). Each source lives in its own module under sources/ and
exposes a top-level `select(filter, n)` function:

    def select(filter: dict, n: int | None = None) -> list[dict]:
        '''Return up to n documents matching `filter`. Each dict has
        keys: doc_id (str), source (str), text (str).'''

To add a new source:
    1. Create <name>.py with a `select()` function.
    2. Add an entry to REGISTRY below.
    3. Reference it from an experiment YAML as
       `corpus.collections[].source: <name>`.

Note on naming: the package is `sources` (not `collections`) to avoid
shadowing Python's stdlib `collections` module. Inside YAMLs we still
say "collections" because that's the natural label for the user-facing
config (we're picking corpus collections to sample from).

Lazy import: `get(name)` only imports the module when actually requested,
so heavy deps (pandas, etc.) don't pay loading cost for unused sources.
"""
from importlib import import_module


# Public name (used in experiment YAMLs) -> module name (file in this dir).
REGISTRY = {
    "bulk_corpus": "bulk_corpus",
    "nyt": "nyt",
    # SWM discourse corpora (LoRA-synth ablation).
    "gst": "gst",                # Congressional speeches (hein bound/daily)
    "economist": "economist",    # The Economist weekly archive
    "fomc": "fomc",              # scraped FOMC minutes/transcripts/statements
    # Future drop-ins (one file each):
    # "ft": "ft",                # D:/.../news_archives/FT/{year}.parquet
    # "newswire": "newswire",    # D:/.../newswire/{year}_data_clean.json
}


def get(name: str):
    """Return the source module for `name`. KeyError if not registered."""
    if name not in REGISTRY:
        raise KeyError(
            f"unknown source: {name!r}. "
            f"Available: {sorted(REGISTRY.keys())}"
        )
    return import_module(f"policy_pred.sources.{REGISTRY[name]}")
