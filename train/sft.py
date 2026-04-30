"""SFT on year-Y synthetic instruction data, init from this year's midtrain checkpoint."""
from pathlib import Path

from .. import config
from ..models.factory import load_backend


def run(year: int, init_spec: dict) -> Path:
    """SFT init_spec on year_sft_path(year).

    Writes to year_checkpoint_dir(year, 'sft') and returns that path.
    """
    backend = load_backend(init_spec)
    return backend.sft(
        sft_path=config.year_sft_path(year),
        output_dir=config.year_checkpoint_dir(year, "sft"),
    )
