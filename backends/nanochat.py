"""nanochat-1900-1949 backend (stub).

Karpathy-style nanochat replication trained on the 1900-1949 era corpus.
Useful as a second base model for methodological robustness on post-1950
policies, and for the chat-format probe Leland asked about (the MSM/SFT
question — see plan).

Not yet implemented. When wiring this up:
- Pull weights from ~/policy_pred_data/models/nanochat_base/ (or wherever the
  Ricardo-replicated checkpoint lives) via config.NANOCHAT_WEIGHTS_DIR.
- Mirror TalkieBackend's surface: _ensure_loaded, _model/_tokenizer/_device,
  score_continuations, generate, prepare_for_peft.
- LORA_TARGET_MODULES should target nanochat's attention/MLP linear submodules
  (different naming convention than Talkie — verify against the checkpoint).
"""
from __future__ import annotations

from pathlib import Path


class NanochatBackend:
    LORA_TARGET_MODULES: list[str] = []  # TODO: populate when implementing

    def __init__(self, weights_dir: Path):
        raise NotImplementedError(
            "NanochatBackend not yet implemented. See backends/nanochat.py "
            "for integration steps, or implement following the TalkieBackend pattern."
        )
