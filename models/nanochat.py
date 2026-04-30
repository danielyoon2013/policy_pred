"""Backend for nanochat checkpoints (the in-tree fork at hist_LLM/nanochat).

Implementations will likely shell out to nanochat's training entrypoints with
configs that point at our corpus / SFT files and init from self.checkpoint_dir.
"""
from pathlib import Path
from typing import Sequence

from .base import ModelBackend


class NanochatBackend(ModelBackend):
    def __init__(self, checkpoint_path):
        self.checkpoint_path = Path(checkpoint_path)

    def midtrain(self, corpus_path: Path, output_dir: Path) -> Path:
        raise NotImplementedError

    def sft(self, sft_path: Path, output_dir: Path) -> Path:
        raise NotImplementedError

    def score_continuations(self, prompt: str, continuations: Sequence[str]) -> list[float]:
        raise NotImplementedError
