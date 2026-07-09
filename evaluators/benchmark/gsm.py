"""GSM-MC evaluator: multi-choice math word problems.

Each test record has fields:
    Question: str           the word problem
    A, B, C, D: str         four candidate answers (numbers as strings)
    Answer: str             the correct letter ("A" | "B" | "C" | "D")

For each problem we build a prompt and score four continuations
(" A", " B", " C", " D") via TalkieBackend.score_continuations.
The highest-logprob option is the predicted answer; accuracy is the
fraction matching the correct letter.

Much faster than open-ended GSM8K: 4 forward passes per problem instead
of N=256 generation steps. ~5,200 forward passes for the full 1,319-problem
test set, vs. ~330K for open-ended.

Config keys (from experiment YAML eval section):
    test_set: str       path to gsm_mc.jsonl (default: D:/hist_LLM/eval_data/gsm_mc.jsonl)
    n_samples: int      cap test items (default: all). Override via --limit too.
    prompt_template: str  override prompt format (default: standard MC format)
"""
from __future__ import annotations

import json
import time
from pathlib import Path


DEFAULT_TEST_SET = Path("D:/hist_LLM/eval_data/gsm_mc.jsonl")

DEFAULT_PROMPT_TEMPLATE = (
    "User: Multiple Choice question: {question}\n"
    "- {a}=A\n"
    "- {b}=B\n"
    "- {c}=C\n"
    "- {d}=D\n"
    "\n"
    "Respond only with the letter of the correct answer.\n"
    "\n"
    "Assistant:"
)

LETTERS = ("A", "B", "C", "D")


def _load_test_set(path: Path) -> list[dict]:
    """Read JSONL, return list of records."""
    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _build_prompt(rec: dict, template: str) -> str:
    return template.format(
        question=rec["Question"].strip(),
        a=str(rec["A"]).strip(),
        b=str(rec["B"]).strip(),
        c=str(rec["C"]).strip(),
        d=str(rec["D"]).strip(),
    )


def run(backend, cfg: dict) -> dict:
    """Run GSM-MC eval against a TalkieBackend (with adapter applied if any).

    Returns a dict ready to JSON-dump with overall accuracy + per-question detail.
    """
    test_path = Path(cfg.get("test_set") or DEFAULT_TEST_SET)
    if not test_path.exists():
        raise FileNotFoundError(
            f"GSM-MC test set not found at {test_path}. "
            f"Set cfg.test_set or place the file at the default path."
        )

    items = _load_test_set(test_path)
    n_samples = cfg.get("n_samples")
    if n_samples is not None:
        items = items[: int(n_samples)]

    prompt_template = cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)

    print(f"GSM-MC eval: {len(items)} problems from {test_path}")
    print(f"  Each problem -> 4 score_continuations calls (" + " ".join(LETTERS) + ")")

    correct = 0
    examples: list[dict] = []
    t0 = time.time()
    options = [f" {L}" for L in LETTERS]  # " A", " B", " C", " D"

    for i, rec in enumerate(items):
        prompt = _build_prompt(rec, prompt_template)
        scores = backend.score_continuations(prompt, options)
        # Pick the option with highest log-prob/token.
        pred_idx = max(range(len(scores)), key=lambda j: scores[j])
        pred_letter = LETTERS[pred_idx]
        gold_letter = str(rec["Answer"]).strip().upper()
        is_correct = pred_letter == gold_letter
        correct += int(is_correct)

        examples.append({
            "question": rec["Question"],
            "options": {L: rec[L] for L in LETTERS},
            "gold": gold_letter,
            "pred": pred_letter,
            "logprobs": dict(zip(LETTERS, scores)),
            "correct": is_correct,
        })

        if (i + 1) % 25 == 0 or i == len(items) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(items)}] correct={correct}  "
                  f"acc={correct/(i+1):.3f}  ({rate:.2f} probs/s)")

    accuracy = correct / len(items) if items else 0.0
    return {
        "metric": "accuracy",
        "pass_at_1": accuracy,        # alias so eval.py's printer recognizes it
        "accuracy": accuracy,
        "n_correct": correct,
        "n_total": len(items),
        "test_set": str(test_path),
        "examples": examples,
    }
