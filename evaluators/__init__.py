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
    "gsm_mc": "gsm_mc",                  # multi-choice math (fast, 4 score_continuations per problem)
    "gsm8k": "gsm8k",                    # open-ended math (generate + parse final number; slow)
    "arc_mc": "arc_mc",                  # multi-choice science reasoning (1144 items, ARC-Challenge)
    "race_mc": "race_mc",                # reading comprehension (RACE-Middle 360 + RACE-High 880)
    "policy_probe": "policy_probe",      # V1 yes/no probe (catalog.yaml only)
    "policy_battery": "policy_battery",  # yes_no + likert5, catalog OR csv (211 events), no year in prompt
    "policy_battery_variants": "policy_battery_variants",  # variant-aware: mean+std over N paraphrases
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
