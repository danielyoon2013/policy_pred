"""Continued pretraining on year-Y raw shard, init from prior year's SFT checkpoint."""
from pathlib import Path

from .. import config
from ..models.factory import load_backend


def run(year: int, init_spec: dict) -> Path:
    """Midtrain init_spec on year_corpus_path(year).

    Writes to year_checkpoint_dir(year, 'midtrain') and returns that path. Backend
    selection is determined by init_spec['type'].
    """
    backend = load_backend(init_spec)
    return backend.midtrain(
        corpus_path=config.year_corpus_path(year),
        output_dir=config.year_checkpoint_dir(year, "midtrain"),
    )
