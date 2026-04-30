"""Smoke test for TalkieBackend on the local Talkie-LM checkpoint.

Validates: ckpt loads, tokenizer works, forward pass produces sane logits,
score_continuations is correct, and the base model's pre-1931 cutoff is
visible at the probe layer (the central project premise).

Run from the policy_pred directory:

    python scripts/smoke_talkie.py

Expect ~10 minutes for the ckpt load on CPU, then a few minutes per probe.
Results print to stdout; nothing is written to disk.
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

from policy_pred.models.factory import load_backend  # noqa: E402


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
    print("Loading talkie_base via registry...")
    print("(53 GB ckpt, mmap streamed -> bf16; first run ~5-10 min on CPU)")
    t0 = time.time()
    backend = load_backend("talkie_base")
    backend._ensure_loaded()  # force the load now so timing isn't mixed in
    print(f"loaded in {time.time() - t0:.1f}s")

    _section("Check 4a: known-fact sanity (Paris vs London)")
    s = _probe(backend, "The capital of France is", [" Paris", " London"])
    print("  PASS" if s[0] > s[1] else "  FAIL", "(expect Paris > London)")

    _section("Check 4b: opposite-sign sanity")
    s = _probe(backend, "The capital of England is", [" Paris", " London"])
    print("  PASS" if s[1] > s[0] else "  FAIL", "(expect London > Paris)")

    _section("Check 5: project-premise probe (the important one)")
    print("  Talkie was trained on text through 1930. Social Security was")
    print("  enacted in 1935. A pre-1931 model should NOT 'know' SS exists.")
    print()
    prompt = (
        "By 1930, the United States had established a national program of "
        "old-age insurance funded by payroll taxes. True or false? Answer:"
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
            "GOOD  — base model does NOT believe SS existed in 1930.\n"
            "         Cutoff is visible at the probe layer.\n"
            "         Project premise holds; proceed to training infra."
        )
    elif delta < -0.5:
        verdict = (
            "CONCERN — base model already strongly believes SS existed in 1930.\n"
            "          Cutoff is NOT visible at the probe layer.\n"
            "          Re-evaluate before spending compute on per-year training."
        )
    else:
        verdict = (
            "AMBIGUOUS — model is roughly 50/50 (|delta| < 0.5 nats).\n"
            "            Consistent with cutoff effect, but inconclusive.\n"
            "            Worth adding more probes before deciding."
        )
    print("  " + verdict.replace("\n", "\n  "))

    print()
    print("Done.")


if __name__ == "__main__":
    main()
