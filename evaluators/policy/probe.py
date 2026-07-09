"""Policy belief-elicitation evaluator (extracted from V1 eval.py).

For a given model + year, runs an "existence as of year Y" Yes/No probe
against each policy in a catalog. Returns log-probabilities + normalized
probs per policy.

Config keys (from experiment YAML eval section):
    catalog: str            path to catalog YAML (default: this dir's catalog.yaml)
    year: int               year used in the probe phrasing
"""
from __future__ import annotations

import math
import time
from pathlib import Path


EVALUATOR_VERSION = "v2.1"


def _load_catalog(path: Path) -> list[dict]:
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["policies"]


def _build_existence_probe(policy: dict, year: int) -> dict:
    description = " ".join(policy["description"].split())  # collapse whitespace
    prompt = (
        f"By {year}, the following exists in the United States or international "
        f"community: {description} True or false? Answer:"
    )
    return {
        "type": "yes_no_existence",
        "prompt": prompt,
        "options": [" True", " False"],
    }


def _score_probe(backend, probe: dict) -> dict:
    scores = backend.score_continuations(probe["prompt"], probe["options"])
    log_probs = list(scores)
    m = max(log_probs)
    exp_shifted = [math.exp(s - m) for s in log_probs]
    total = sum(exp_shifted)
    return {
        **probe,
        "scores": log_probs,
        "p_normalized": [v / total for v in exp_shifted],
    }


def run(backend, cfg: dict) -> dict:
    """Run the policy-probe eval against a TalkieBackend (with adapter applied)."""
    default_catalog = Path(__file__).parent / "catalog.yaml"
    catalog_path = Path(cfg.get("catalog", default_catalog))
    year = cfg["year"]

    policies = _load_catalog(catalog_path)
    print(f"  catalog: {len(policies)} policies from {catalog_path}")

    results: list[dict] = []
    for policy in policies:
        t0 = time.time()
        probe = _build_existence_probe(policy, year)
        scored = _score_probe(backend, probe)
        s_true, s_false = scored["scores"]
        p_true, _ = scored["p_normalized"]
        print(f"  {policy['id']:<26s}  True {s_true:+7.3f} / "
              f"False {s_false:+7.3f}  P(True)={p_true:.3f}")
        results.append({
            "policy_id": policy["id"],
            "implementation_year": policy.get("implementation_year"),
            "anticipation_start_year": policy.get("anticipation_start_year"),
            "probe": scored,
            "elapsed_sec": time.time() - t0,
        })

    return {
        "evaluator": "policy_probe",
        "evaluator_version": EVALUATOR_VERSION,
        "year": year,
        "n_policies": len(policies),
        "policies": results,
    }
