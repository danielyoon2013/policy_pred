"""ARC Challenge multi-choice evaluator.

Same shape as gsm_mc but for the AllenAI ARC-Challenge science-reasoning
benchmark. Each test record has fields:
    Question: str           the science question
    A, B, C, D: str         four candidate answers
    Answer: str             the correct letter ("A" | "B" | "C" | "D")

For each problem we score four continuations (" A", " B", " C", " D")
via TalkieBackend.score_continuations and pick the highest-logprob letter.

ARC-Challenge has 1172 test items; 1144 of them have exactly 4 choices.
The conversion script that produced arc_challenge_test.jsonl drops the
remaining 28 (3-choice / 5-choice / non-A-D-labeled).

Config keys (from experiment YAML eval section):
    test_set: str       path to arc_challenge_test.jsonl
    n_samples: int      cap test items (default: all)
    prompt_template: str  override prompt format
"""
from __future__ import annotations

import json
import time
from pathlib import Path


DEFAULT_TEST_SET = Path("D:/hist_LLM/eval_data/arc_challenge_test.jsonl")

DEFAULT_PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "\n"
    "A. {a}\n"
    "B. {b}\n"
    "C. {c}\n"
    "D. {d}\n"
    "\n"
    "Answer:"
)

LETTERS = ("A", "B", "C", "D")


def _load_test_set(path: Path) -> list[dict]:
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
    test_path = Path(cfg.get("test_set") or DEFAULT_TEST_SET)
    if not test_path.exists():
        raise FileNotFoundError(
            f"ARC-Challenge test set not found at {test_path}. "
            f"Set cfg.test_set or place the file at the default path."
        )

    items = _load_test_set(test_path)
    n_samples = cfg.get("n_samples")
    if n_samples is not None:
        items = items[: int(n_samples)]

    prompt_template = cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)

    print(f"ARC-Challenge eval: {len(items)} problems from {test_path}")
    print(f"  Each problem -> 4 score_continuations calls (" + " ".join(LETTERS) + ")")

    correct = 0
    examples: list[dict] = []
    t0 = time.time()
    options = [f" {L}" for L in LETTERS]

    for i, rec in enumerate(items):
        prompt = _build_prompt(rec, prompt_template)
        scores = backend.score_continuations(prompt, options)
        pred_idx = max(range(len(scores)), key=lambda j: scores[j])
        pred_letter = LETTERS[pred_idx]
        gold_letter = str(rec["Answer"]).strip().upper()
        is_correct = pred_letter == gold_letter
        correct += int(is_correct)

        examples.append({
            "id": rec.get("id"),
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
        "pass_at_1": accuracy,
        "accuracy": accuracy,
        "n_correct": correct,
        "n_total": len(items),
        "test_set": str(test_path),
        "examples": examples,
    }
