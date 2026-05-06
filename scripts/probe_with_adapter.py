"""Probe base Talkie and (optionally) base + LoRA adapter; show side-by-side.

Used after train.py to validate that the trained LoRA adapter actually
shifts the model's output distribution. Runs three probes:
  - Two sanity probes (Paris/London) — should still pass after training.
  - The Social Security 1930 probe — should shift if the adapter learned
    something about 1931 (or any post-1930 era text).

Usage:
    python scripts/probe_with_adapter.py                      # base only
    python scripts/probe_with_adapter.py --adapter <dir>      # base vs base+adapter

Run on remote (CUDA) for speed; CPU works but is slow.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make `policy_pred.X` imports work when run from inside the package directory.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO.parent))

from policy_pred import config  # noqa: E402
from policy_pred.talkie import TalkieBackend  # noqa: E402


PROBES = [
    (
        "sanity-fr",
        "The capital of France is",
        [" Paris", " London"],
        "expect Paris > London",
    ),
    (
        "sanity-en",
        "The capital of England is",
        [" Paris", " London"],
        "expect London > Paris",
    ),
    (
        "ss-1930",
        "By 1930, the United States had established a national program of "
        "old-age insurance funded by payroll taxes. True or false? Answer:",
        [" True", " False"],
        "post-1930 probe; values should change after training",
    ),
]


def run_probes(backend: TalkieBackend) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for name, prompt, opts, hint in PROBES:
        t0 = time.time()
        scores = backend.score_continuations(prompt, opts)
        out[name] = scores
        print(f"  [{name}]  ({hint})")
        for opt, s in zip(opts, scores):
            print(f"      {opt!r:>10s}  {s:+.4f}")
        print(f"      ({time.time() - t0:.1f}s)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path,
                   help="Directory containing PEFT adapter (adapter_config.json + weights)")
    args = p.parse_args()

    print(f"Loading Talkie base from {config.TALKIE_WEIGHTS_DIR}...")
    backend = TalkieBackend(config.TALKIE_WEIGHTS_DIR)
    backend._ensure_loaded()

    print("\n=== BASE ===")
    base_scores = run_probes(backend)

    if args.adapter is None:
        print("\n(no --adapter given; base-only run complete.)")
        return

    if not args.adapter.exists():
        sys.exit(f"adapter directory not found: {args.adapter}")

    print(f"\nLoading adapter from {args.adapter}...")
    from peft import PeftModel
    backend._model = PeftModel.from_pretrained(backend._model, str(args.adapter))
    backend._model.eval()

    print("\n=== BASE + ADAPTER ===")
    adapter_scores = run_probes(backend)

    print("\n=== COMPARISON ===")
    for name, prompt, opts, _ in PROBES:
        print(f"\n[{name}]  {prompt!r}")
        for opt, b, a in zip(opts, base_scores[name], adapter_scores[name]):
            delta = a - b
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "·"
            print(
                f"    {opt!r:>10s}  base {b:+.4f}  +adapter {a:+.4f}  "
                f"Δ {delta:+.4f} {arrow}"
            )

    # Summary signal: did the adapter move SS-1930 at all?
    name = "ss-1930"
    deltas = [
        a - b
        for b, a in zip(base_scores[name], adapter_scores[name])
    ]
    abs_max = max(abs(d) for d in deltas)
    print(
        f"\nSS-1930 max |Δ|: {abs_max:.4f} nats. "
        + ("training appears to have moved the distribution."
           if abs_max > 0.05
           else "training had little measurable effect on this probe.")
    )


if __name__ == "__main__":
    main()
