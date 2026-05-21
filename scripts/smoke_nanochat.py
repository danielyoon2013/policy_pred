"""Smoke test for NanochatBackend on the 1900-1949 base checkpoint.

Validates: vendor source loads, ckpt + tokenizer load, forward pass
produces sane logits, score_continuations is correct, and the base
model's era cutoff is visible at the probe layer (same sanity check
as smoke_talkie.py but anchored at the 1949 cutoff instead of 1930).

Run from the policy_pred directory:

    python scripts/smoke_nanochat.py

~30s to 1min on CPU for the much smaller nanochat ckpt (~4 GB vs Talkie's
53 GB). On GPU it's faster.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

# Make `policy_pred.X` imports work when run from inside the package directory.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO.parent))

from policy_pred import config  # noqa: E402
from policy_pred.backends.nanochat import NanochatBackend  # noqa: E402


def _section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def _probe(backend, prompt: str, options: list[str]) -> list[float]:
    print(f"  prompt:  {prompt!r}")
    t0 = time.time()
    scores = backend.score_continuations(prompt, options)
    dt = time.time() - t0
    for opt, s in zip(options, scores):
        print(f"    {opt!r:>10s}  log-p/tok = {s:+.4f}")
    print(f"  ({dt:.1f}s for {len(options)} continuations)")
    return scores


def main() -> None:
    print(f"Loading nanochat from {config.NANOCHAT_WEIGHTS_DIR}...")
    print("(~4 GB ckpt; ~30s-1min on CPU, faster on GPU)")
    t0 = time.time()
    backend = NanochatBackend(config.NANOCHAT_WEIGHTS_DIR)
    backend._ensure_loaded()
    print(f"loaded in {time.time() - t0:.1f}s")
    print(f"device:   {backend._device}")
    print(f"model:    {type(backend._model).__name__}, "
          f"{sum(p.numel() for p in backend._model.parameters())/1e6:.1f}M params")

    _section("Check 1: known-fact sanity (Paris vs London)")
    s = _probe(backend, "The capital of France is", [" Paris", " London"])
    print("  PASS" if s[0] > s[1] else "  FAIL", "(expect Paris > London)")

    _section("Check 2: opposite-sign sanity")
    s = _probe(backend, "The capital of England is", [" Paris", " London"])
    print("  PASS" if s[1] > s[0] else "  FAIL", "(expect London > Paris)")

    _section("Check 3: project-premise probe (era-cutoff visibility)")
    print("  Nanochat was trained on 1900-1949 text. The Civil Rights Act")
    print("  was enacted in 1964 — well after the cutoff. A pre-1950 model")
    print("  should NOT 'know' it exists.")
    print()
    prompt = (
        "By 1949, the United States had enacted a federal Civil Rights Act "
        "outlawing racial segregation in public accommodations. True or "
        "false? Answer:"
    )
    s = _probe(backend, prompt, [" True", " False"])

    p_t = math.exp(s[0])
    p_f = math.exp(s[1])
    norm = p_t + p_f
    print(f"  normalised: P(True)={p_t/norm:.3f}  P(False)={p_f/norm:.3f}")
    print()

    delta = s[1] - s[0]  # positive if False > True
    if delta > 0.5:
        verdict = (
            "GOOD  — base model does NOT believe Civil Rights Act existed by 1949.\n"
            "         Cutoff is visible at the probe layer."
        )
    elif delta < -0.5:
        verdict = (
            "CONCERN — base model strongly believes Civil Rights Act existed by 1949.\n"
            "          Cutoff is NOT visible at the probe layer."
        )
    else:
        verdict = (
            "AMBIGUOUS — model is roughly 50/50 (|delta| < 0.5 nats).\n"
            "            Consistent with cutoff effect, but inconclusive."
        )
    print("  " + verdict.replace("\n", "\n  "))

    print()
    print("Done.")


if __name__ == "__main__":
    main()
