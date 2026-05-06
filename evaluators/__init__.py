"""Plug-in registry for evaluation methods.

An "evaluator" runs a model against a benchmark and returns metrics. Each
evaluator lives in its own module under evaluators/ and exposes a top-level
`run(model, tokenizer, cfg)` function:

    def run(model, tokenizer, cfg: dict) -> dict:
        '''Return a dict of metrics + per-example details.'''

To add a new evaluator:
    1. Create <name>.py with a `run()` function.
    2. Add an entry to REGISTRY below.
    3. Reference it from an experiment YAML as
       `eval.evaluator: <name>`.

Currently only `gsm8k` is implemented. `policy_probe` will be added when
we move to the policy V2 work (it'll wrap the existing eval.py logic).
"""
from importlib import import_module


# Public name (used in experiment YAMLs) -> module name (file in this dir).
REGISTRY = {
    "gsm8k": "gsm8k",
    "policy_probe": "policy_probe",
    # Future drop-ins:
    # "humaneval": "humaneval",         # cheap general-cap sanity
}


def get(name: str):
    """Return the evaluator module for `name`. KeyError if not registered."""
    if name not in REGISTRY:
        raise KeyError(
            f"unknown evaluator: {name!r}. "
            f"Available: {sorted(REGISTRY.keys())}"
        )
    return import_module(f"policy_pred.evaluators.{REGISTRY[name]}")
