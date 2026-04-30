"""Construct a ModelBackend from a small dict spec.

A spec carries 'type' (which backend) and 'path' (where to load from). Any extra
keys are forwarded to the backend constructor as keyword arguments, so backend-
specific options (LR overrides, dtype, etc.) live in the same dict the pipeline
already passes around.
"""
from .base import ModelBackend


def load_backend(spec_or_name) -> ModelBackend:
    """Load a backend from either a registry name or a full spec dict.

    str  -> looked up via models/registry.yaml (use this for the BASE_MODEL).
    dict -> used directly (use this when the path is dynamic, e.g. a year-Y
            checkpoint produced by an earlier pipeline stage).
    """
    if isinstance(spec_or_name, str):
        from .registry import resolve
        spec = resolve(spec_or_name)
    else:
        spec = spec_or_name
    kind = spec["type"]
    extras = {k: v for k, v in spec.items() if k not in {"type", "path"}}
    if kind == "nanochat":
        from .nanochat import NanochatBackend
        return NanochatBackend(spec["path"], **extras)
    if kind == "talkie":
        from .talkie import TalkieBackend
        return TalkieBackend(spec["path"], **extras)
    raise ValueError(f"unknown backend type: {kind!r}")
