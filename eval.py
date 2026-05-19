"""V2 stage 4 (h): evaluate a trained model via plug-in evaluators.

Reads experiment YAML, loads M_base + experiment's adapter (if present),
dispatches to evaluators[<name>].run(backend, cfg), writes A.json to
D:/.../experiments/<name>/eval.json.

Run:
    python eval.py --experiment experiments/math_lora_r32.yaml

Idempotent: skips if eval.json exists, --force to overwrite. Pass
--no-adapter to evaluate the base model alone (useful for comparison
baselines).

The legacy `python eval.py --year YYYY` CLI from V1 is no longer
supported here; that flow is now reachable as a `policy_probe` evaluator
via an experiment YAML.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make policy_pred.* importable when run as a script from the repo root.
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from policy_pred import config, evaluators  # noqa: E402
from policy_pred.talkie import TalkieBackend  # noqa: E402
from policy_pred.corpus import experiment_dir, load_experiment  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True,
                   help="Path to experiment YAML.")
    p.add_argument("--no-adapter", action="store_true",
                   help="Evaluate base only, regardless of cfg.")
    p.add_argument("--adapter", type=Path, default=None,
                   help="Override the auto-resolved adapter path (which is "
                        "experiment_dir(name)/checkpoint). Use when two eval "
                        "YAMLs share one trained adapter from a different "
                        "experiment name.")
    p.add_argument("--evaluator", type=str, default=None,
                   help="Override cfg.eval.evaluator from the YAML. Use to "
                        "rerun an existing experiment with a different "
                        "evaluator without editing the YAML (e.g. "
                        "policy_battery -> policy_battery_variants).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing eval.json.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment(args.experiment)
    name = cfg["name"]
    out_dir = experiment_dir(name)
    out_path = out_dir / "eval.json"

    if out_path.exists() and not args.force:
        print(f"eval.json exists at {out_path} (--force to overwrite)")
        return

    # Load backend.
    print(f"Loading Talkie from {config.TALKIE_WEIGHTS_DIR}...")
    backend = TalkieBackend(config.TALKIE_WEIGHTS_DIR)
    backend._ensure_loaded()

    # Apply adapter if available and not suppressed. --adapter overrides
    # the default name-based lookup; useful when two eval YAMLs share one
    # trained adapter (e.g. factmath_eval_arc + factmath_eval_gsm both
    # pointing at experiments/factmath_sft/checkpoint).
    if args.no_adapter:
        adapter_dir = None
    elif args.adapter is not None:
        adapter_dir = args.adapter
    else:
        adapter_dir = out_dir / "checkpoint"
    if adapter_dir is not None and adapter_dir.exists():
        print(f"Applying adapter {adapter_dir}...")
        from peft import PeftModel
        # peft >=0.10 calls model.config.get("tie_word_embeddings") inside
        # inject_adapter. Talkie's GPTConfig is a plain dataclass-like object
        # without .get(), so shim it before peft touches it.
        _model_cfg = getattr(backend._model, "config", None)
        if _model_cfg is not None and not hasattr(_model_cfg, "get"):
            _model_cfg.get = lambda k, default=None: getattr(_model_cfg, k, default)
        backend._model = PeftModel.from_pretrained(backend._model, str(adapter_dir))
        backend._model.eval()
        adapter_label = str(adapter_dir)
    else:
        if adapter_dir is not None:
            print(f"Note: no adapter at {adapter_dir}; running base-only eval.")
        else:
            print("--no-adapter: base-only eval.")
        adapter_label = None

    # Dispatch to the configured evaluator (CLI --evaluator overrides YAML).
    eval_cfg = dict(cfg["eval"])  # copy so we can write into it
    # Surface the experiment name to evaluators so they can derive the
    # year-model index for lookback-window filtering.
    eval_cfg["_experiment_name"] = name
    evaluator_name = args.evaluator or eval_cfg["evaluator"]
    if args.evaluator and args.evaluator != eval_cfg.get("evaluator"):
        print(f"--evaluator override: {eval_cfg.get('evaluator')!r} -> {args.evaluator!r}")
    evaluator = evaluators.get(evaluator_name)
    print(f"\nRunning evaluator '{evaluator_name}'...")
    t0 = time.time()
    results = evaluator.run(backend, eval_cfg)
    elapsed = time.time() - t0

    out = {
        "experiment": name,
        "evaluator": evaluator_name,
        "adapter": adapter_label,
        "talkie_weights_dir": str(config.TALKIE_WEIGHTS_DIR),
        "elapsed_sec": elapsed,
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Print headline metric if recognizable.
    if "pass_at_1" in results:
        print(f"\npass@1 = {results['pass_at_1']:.3f} "
              f"({results['n_correct']}/{results['n_total']})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
