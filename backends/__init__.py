"""Dispatcher: pick the right backend class for an experiment config.

Reads `cfg["model_type"]` (default "talkie"), maps it to the matching
weights dir from policy_pred.config, lazy-imports the backend module so
heavy deps (transformers, peft) aren't pulled in for the Talkie-only
code path, and returns a fresh backend instance.
"""
from __future__ import annotations

from .base import Backend


def load_backend(cfg: dict) -> Backend:
    """Construct a Backend from an experiment config dict."""
    from policy_pred import config

    mt = (cfg.get("model_type") or "talkie").lower()
    if mt == "talkie":
        from .talkie import TalkieBackend
        return TalkieBackend(config.TALKIE_WEIGHTS_DIR)
    if mt == "nanochat":
        from .nanochat import NanochatBackend
        return NanochatBackend(config.NANOCHAT_WEIGHTS_DIR)
    if mt == "qwen":
        from .qwen import QwenBackend
        return QwenBackend(config.QWEN_WEIGHTS_DIR)
    raise ValueError(
        f"unknown model_type: {mt!r} (expected 'talkie' | 'nanochat' | 'qwen')"
    )


__all__ = ["Backend", "load_backend"]
