"""V2 step h: evaluate(M_t, Q_t) -> A_t.

For a year-Y model (M_base + optional adapter), score every policy in the
catalog and write one JSON file per (policy, year) under eval/{policy_id}/{Y}.json.

V2.1 probe form: a single "existence as of year Y" Yes/No probe per policy,
using catalog.description as the paraphrase. Future revisions will add
additional probe forms (year-completion, MC over years, paraphrase batteries)
without changing the on-disk schema (`probes` is a list).

Usage:
    # Year 1931 with the year's adapter (default):
    python evaluate.py --year 1931

    # Year 1931, base only (no adapter):
    python evaluate.py --year 1931 --no-adapter

    # Year 1931, adapter from a non-default location:
    python evaluate.py --year 1931 --adapter checkpoints/test_run

CPU is fine; this is just inference. For quick comparison-style runs against
a single adapter, scripts/probe_with_adapter.py is faster (no per-policy I/O).
This script is the durable, persisted eval.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Make policy_pred.* importable when run as a script from the repo root.
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from policy_pred import config  # noqa: E402
from policy_pred.talkie import TalkieBackend  # noqa: E402


EVALUATOR_VERSION = "v2.1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", type=int, required=True,
                   help="Year of the model being evaluated; used for the probe "
                        "phrasing AND to locate the default adapter path.")
    p.add_argument("--adapter", type=Path, default=None,
                   help="Override adapter path. Default: year_checkpoint_dir(Y).")
    p.add_argument("--no-adapter", action="store_true",
                   help="Evaluate base only (no LoRA adapter).")
    p.add_argument("--catalog", type=Path, default=config.POLICY_CATALOG_PATH,
                   help=f"Catalog YAML (default {config.POLICY_CATALOG_PATH}).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing eval files for this year.")
    return p.parse_args()


def load_catalog(path: Path) -> list[dict]:
    """Read the catalog YAML and return a list of policy dicts."""
    import yaml
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["policies"]


def build_existence_probe(policy: dict, year: int) -> dict:
    """Build the V2.1 'existence as of year Y' Yes/No probe.

    Uses the paraphrased catalog description (NOT the act name) so this is a
    belief-elicitation, not a verbatim-recall test.
    """
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


def score_probe(backend: TalkieBackend, probe: dict) -> dict:
    """Run score_continuations and add normalized probabilities."""
    scores = backend.score_continuations(probe["prompt"], probe["options"])
    log_probs = list(scores)
    # Normalize to a probability distribution over the options.
    # log-sum-exp for numerical stability.
    m = max(log_probs)
    exp_shifted = [math.exp(s - m) for s in log_probs]
    total = sum(exp_shifted)
    p_norm = [v / total for v in exp_shifted]
    return {
        **probe,
        "scores": log_probs,
        "p_normalized": p_norm,
    }


def main() -> None:
    args = parse_args()

    # Resolve adapter path.
    if args.no_adapter:
        adapter_dir = None
    elif args.adapter is not None:
        adapter_dir = args.adapter
    else:
        adapter_dir = config.year_checkpoint_dir(args.year)
        if not adapter_dir.exists():
            print(f"Note: default adapter dir {adapter_dir} not found; "
                  f"falling back to base-only eval. Pass --no-adapter to "
                  f"silence this notice or --adapter to override.")
            adapter_dir = None

    # Load catalog.
    policies = load_catalog(args.catalog)
    print(f"Catalog: {len(policies)} policies from {args.catalog}")

    # Check which (policy, year) results already exist; skip unless --force.
    pending: list[dict] = []
    for policy in policies:
        out_path = config.policy_eval_path(args.year, policy["id"])
        if out_path.exists() and not args.force:
            print(f"  skip {policy['id']} ({out_path} exists; use --force to overwrite)")
            continue
        pending.append(policy)
    if not pending:
        print("All policies already evaluated for this year. Nothing to do.")
        return

    # Load model.
    print(f"\nLoading Talkie base from {config.TALKIE_WEIGHTS_DIR}...")
    backend = TalkieBackend(config.TALKIE_WEIGHTS_DIR)
    backend._ensure_loaded()

    if adapter_dir is not None:
        print(f"Applying adapter {adapter_dir}...")
        from peft import PeftModel
        backend._model = PeftModel.from_pretrained(
            backend._model, str(adapter_dir)
        )
        backend._model.eval()
        adapter_label = str(adapter_dir)
    else:
        print("No adapter (base-only eval).")
        adapter_label = None

    # Run each policy.
    print(f"\nRunning {len(pending)} probes for year {args.year}...")
    for policy in pending:
        t0 = time.time()
        probe = build_existence_probe(policy, args.year)
        scored = score_probe(backend, probe)

        out: dict = {
            "policy_id": policy["id"],
            "year": args.year,
            "implementation_year": policy.get("implementation_year"),
            "anticipation_start_year": policy.get("anticipation_start_year"),
            "adapter": adapter_label,
            "probes": [scored],
            "metadata": {
                "evaluator_version": EVALUATOR_VERSION,
                "talkie_weights_dir": str(config.TALKIE_WEIGHTS_DIR),
                "elapsed_sec": time.time() - t0,
            },
        }
        out_path = config.policy_eval_path(args.year, policy["id"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        # Print a one-liner per policy for visual scan.
        s_true, s_false = scored["scores"]
        p_true, p_false = scored["p_normalized"]
        print(
            f"  {policy['id']:<26s}  "
            f"True {s_true:+7.3f} / False {s_false:+7.3f}  "
            f"P(True)={p_true:.3f}  "
            f"-> {out_path.relative_to(config.DATA_ROOT) if str(out_path).startswith(str(config.DATA_ROOT)) else out_path}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
