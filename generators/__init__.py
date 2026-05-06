"""Plug-in registry for synthetic-data generators.

A "generator" turns seed documents into training-format JSONL. Each
generator lives in its own module under generators/ and exposes a class
implementing the abstract Generator interface (see base.py):

    class FooGenerator(Generator):
        def generate(self, seeds: list[dict], cfg: dict) -> list[dict]:
            '''Return a list of {"messages": [...]} chat-format records.'''

To add a new generator:
    1. Create <name>.py with a class subclassing Generator from base.py.
    2. Add prompts to prompts/<name>_system.md and prompts/<name>_user.md.
    3. Add an entry to REGISTRY below.
    4. Reference it from an experiment YAML as
       `synth.generators.<name>: { ... per-generator cfg ... }`.

Currently only `math_word` is implemented; we add more (factual, cot,
rephrase, era_paraphrase) as the policy V2 work needs them.
"""
from importlib import import_module


# Public name (used in experiment YAMLs) -> "module:Class".
REGISTRY = {
    "math_word": "math_word:MathWordGenerator",
}


def get(name: str):
    """Return the Generator class for `name`. KeyError if not registered."""
    if name not in REGISTRY:
        raise KeyError(
            f"unknown generator: {name!r}. "
            f"Available: {sorted(REGISTRY.keys())}"
        )
    module_name, class_name = REGISTRY[name].split(":")
    module = import_module(f"policy_pred.generators.{module_name}")
    return getattr(module, class_name)
