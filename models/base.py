"""Backend-agnostic interface for the year-by-year pipeline.

A 'backend' wraps whatever library actually loads the weights, runs training, and
scores tokens -- nanochat for our in-tree fork, HF transformers for Talkie-LM, or
anything we add later. The pipeline interacts only with the methods on this class
so adding a new backend means writing one file, not threading conditionals through
the whole codebase.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence


class ModelBackend(ABC):
    @abstractmethod
    def midtrain(self, corpus_path: Path, output_dir: Path) -> Path:
        """Continued pretraining on raw text at corpus_path. Return path to new ckpt."""
        ...

    @abstractmethod
    def sft(self, sft_path: Path, output_dir: Path) -> Path:
        """SFT on JSONL instructions at sft_path. Return path to new ckpt."""
        ...

    @abstractmethod
    def score_continuations(
        self, prompt: str, continuations: Sequence[str]
    ) -> list[float]:
        """Length-normalized log-prob of each continuation given prompt.

        Used for both MC probes (one continuation per option) and Yes/No questions
        (continuations=['Yes', 'No']). Length-normalization (divide by token count)
        prevents bias toward shorter answers.
        """
        ...
