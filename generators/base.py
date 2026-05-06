"""Abstract base class for synthetic-data generators.

Each concrete generator subclasses Generator and implements `generate()`,
which takes seed documents (W) and returns chat-format training records
(part of S). Generators run synchronously by default; we'll add an
OpenAI Batch API path later for cost savings on large runs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable


class Generator(ABC):
    """Each generator turns seed docs into chat-format training records.

    Subclasses configure prompt paths and any per-generator hyperparameters
    in __init__, then implement `generate()`.
    """

    name: str  # set by subclass; matches the registry key

    def __init__(self, cfg: dict):
        """cfg is the per-generator dict from the experiment YAML.

        Standard keys (subclasses can read more):
            n_per_seed: int     how many output records per seed doc
            format: str         "open" | "cot" | "mc4" | ...
            openai: dict        nested OpenAI API config (model, etc.)
        """
        self.cfg = cfg
        self.n_per_seed = cfg.get("n_per_seed", 1)
        self.format = cfg.get("format", "open")

    @abstractmethod
    def generate(self, seeds: Iterable[dict]) -> list[dict]:
        """Run the generator over seed docs and return list of chat records.

        Each output record is {"messages": [{"role", "content"}, ...]}.
        Implementations should be deterministic given (seeds, cfg, rng_seed)
        for reproducibility.
        """
        ...

    def load_prompt(self, suffix: str) -> str:
        """Read prompts/<name>_<suffix>.md from this package.

        E.g. self.load_prompt("system") returns prompts/<name>_system.md.
        """
        prompt_path = Path(__file__).parent / "prompts" / f"{self.name}_{suffix}.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"missing prompt: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")
