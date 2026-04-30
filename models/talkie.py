"""Backend for Talkie-LM (Hugging Face causal LM).

Accepts either an HF Hub id ('talkie-lm/...') or a local path to a saved model.
Implementations will likely use transformers Trainer (or a small custom loop)
with model.save_pretrained(output_dir) for checkpoint output.
"""
from pathlib import Path
from typing import Sequence

from .base import ModelBackend


class TalkieBackend(ModelBackend):
    def __init__(self, source):
        # HF Hub id (str) or filesystem path -- transformers accepts both.
        self.source = str(source)

    def midtrain(self, corpus_path: Path, output_dir: Path) -> Path:
        raise NotImplementedError

    def sft(self, sft_path: Path, output_dir: Path) -> Path:
        raise NotImplementedError

    def score_continuations(self, prompt: str, continuations: Sequence[str]) -> list[float]:
        raise NotImplementedError
