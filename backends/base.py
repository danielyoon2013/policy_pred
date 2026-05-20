"""Backend Protocol — the surface every base-model backend must expose.

train.py + eval.py + the evaluators all see backends through this duck-typed
interface (PEP 544 Protocol). Adding a new backend means dropping a new file
into this package, satisfying the protocol, and wiring it into the dispatcher
in `__init__.py`. No inheritance needed.

A backend's job is: lazy-load a base LM + tokenizer, expose them, and provide
log-prob scoring of continuations (the eval primitive) and greedy generation.
Architecture-specific knowledge (LoRA target module names, peft-compat shims)
lives with the backend, not in the orchestration layer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import torch


@runtime_checkable
class Backend(Protocol):
    weights_dir: Path
    LORA_TARGET_MODULES: Sequence[str]

    # Attribute access. _model is mutated externally when peft adapters are
    # attached (eval.py, probe_with_adapter.py), so it must be writable.
    _model: torch.nn.Module
    _tokenizer: object
    _device: torch.device

    def _ensure_loaded(self) -> None: ...

    def score_continuations(
        self, prompt: str, continuations: Sequence[str]
    ) -> list[float]: ...

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        stop_sequences: Sequence[str] = (),
    ) -> str: ...

    def prepare_for_peft(self) -> None:
        """Optional hook for per-backend peft compatibility shims.

        Called by train.py / eval.py right before any peft call. Default is
        a no-op; backends with non-HF model configs (e.g. Talkie's dataclass
        GPTConfig) override this to make their config look dict-like.
        """
        ...
