"""Plug-in registry for evaluation methods.

Each "evaluator" runs a model against a benchmark and returns metrics. Evaluators
live under evaluators/<family>/ and expose a top-level `run(backend, cfg) -> dict`:

    def run(backend, cfg: dict) -> dict:
        '''Return a dict of metrics + per-example details.'''

Families (one folder per test the professor cares about):
    policy/     (1) policy prediction  — belief probes over the policy-event battery
    value/      (2) value assessment   — model-as-survey-respondent, P(focal answer)
    benchmark/  (3) LAB look-ahead + (4) external capability — MC via score_continuations

To add an evaluator: create <family>/<name>.py with a `run()`, add an entry to
REGISTRY below (public name -> dotted module under evaluators/), and reference it
from an experiment YAML as `eval.evaluator: <public name>`.
"""
from importlib import import_module


# Public name (used in experiment YAMLs) -> dotted module path under evaluators/.
# Public names are STABLE — experiment YAMLs depend on them; only the module
# location changed when evaluators were grouped into family folders.
REGISTRY = {
    # (1) policy prediction
    "policy_probe": "policy.probe",                        # V1 yes/no probe (catalog.yaml only)
    "policy_battery": "policy.battery",                    # yes_no + likert5, catalog OR csv (211 events)
    "policy_battery_variants": "policy.battery_variants",  # variant-aware: mean+std over N paraphrases
    # (2) value assessment
    "values_battery": "value.survey",                      # survey items as respondent probes, P(focal answer)
    # (3) LAB + (4) external benchmarks (multiple-choice via score_continuations)
    "arc_mc": "benchmark.arc",                             # ARC-Challenge science reasoning
    "gsm_mc": "benchmark.gsm",                             # multi-choice math (fast)
    "gsm8k": "benchmark.gsm8k",                            # open-ended math (generate + parse)
    "race_mc": "benchmark.race",                           # reading comprehension (RACE middle+high)
    "nanochat_task": "benchmark.nanochat_task",            # bridge: LAB + HellaSwag/Winogrande/PIQA/ARC/MMLU[-History]
}


def get(name: str):
    """Return the evaluator module for `name`. KeyError if not registered."""
    if name not in REGISTRY:
        raise KeyError(
            f"unknown evaluator: {name!r}. "
            f"Available: {sorted(REGISTRY.keys())}"
        )
    return import_module(f"policy_pred.evaluators.{REGISTRY[name]}")
