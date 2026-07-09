"""Dispatcher: pick the right backend class for an experiment config.

Reads `cfg["model_type"]` (default "talkie"), maps it to the matching
weights dir from policy_pred.config, lazy-imports the backend module so
heavy deps (transformers, peft) aren't pulled in for the Talkie-only
code path, and returns a fresh backend instance.
"""
from __future__ import annotations

from pathlib import Path

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
        # Optional `checkpoint:` block pins a specific vintage instead of the
        # default (latest base checkpoint):
        #   checkpoint: {year: 1955}    -> continual d23 vintage for that year
        #   checkpoint: {weights_dir, subdir, model_tag, step, tokenizer_dir}
        ck = cfg.get("checkpoint") or {}
        if ck.get("year") is not None:
            tag = ck.get("model_tag", "d23")
            return NanochatBackend(
                config.CONTINUAL_WEIGHTS_DIR,
                checkpoint_subdir=ck.get("subdir", "year_checkpoints"),
                model_tag=tag,
                step=config.year_to_step(int(ck["year"]), tag),
                tokenizer_dir=config.CONTINUAL_TOKENIZER_DIR,
            )
        if ck:
            return NanochatBackend(
                Path(ck["weights_dir"]) if ck.get("weights_dir") else config.NANOCHAT_WEIGHTS_DIR,
                checkpoint_subdir=ck.get("subdir", "base_checkpoints"),
                model_tag=ck.get("model_tag"),
                step=ck.get("step"),
                tokenizer_dir=Path(ck["tokenizer_dir"]) if ck.get("tokenizer_dir") else None,
            )
        return NanochatBackend(config.NANOCHAT_WEIGHTS_DIR)
    if mt == "qwen":
        from .qwen import QwenBackend
        return QwenBackend(config.QWEN_WEIGHTS_DIR)
    raise ValueError(
        f"unknown model_type: {mt!r} (expected 'talkie' | 'nanochat' | 'qwen')"
    )


__all__ = ["Backend", "load_backend"]
