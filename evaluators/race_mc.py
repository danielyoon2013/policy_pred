"""RACE-Middle + RACE-High multi-choice reading comprehension evaluator.

Same shape as arc_mc but with a different record schema. Each test record is:
    question: str       the FULL user prompt, already pre-rendered as
                        "Multiple Choice question: Read the following passage
                        and answer the question.\\n\\nPassage: ...\\n\\n
                        {the actual question}\\n- choice=A\\n- choice=B\\n...
                        \\n\\nRespond only with the letter of the correct answer."
    expected: str       the correct letter ("A" | "B" | "C" | "D")
    subset: str         "middle" | "high" (informational only)

Test data extracted locally from hist_llm's prior error_analysis output (the
`question` and `expected` fields), avoiding a fresh HF download.

For each item we just prepend "User: " and append "\\n\\nAssistant:" to the
question text, then score the four " A"/" B"/" C"/" D" continuations.

Config keys (from experiment YAML eval section):
    test_set: str       path to race_test.jsonl (must have question + expected)
    n_samples: int      cap test items (default: all)
    subset_filter: str  "middle" | "high" | "all" (default: "all")
"""
from __future__ import annotations

import json
import time
from pathlib import Path


DEFAULT_TEST_SET = Path("data_links/eval/race_test.jsonl")
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


def run(backend, cfg: dict) -> dict:
    test_path = Path(cfg.get("test_set") or DEFAULT_TEST_SET)
    if not test_path.exists():
        raise FileNotFoundError(
            f"RACE test set not found at {test_path}. "
            f"Set cfg.test_set or place the file at the default path."
        )

    items = _load_test_set(test_path)

    subset_filter = (cfg.get("subset_filter") or "all").lower()
    if subset_filter in ("middle", "high"):
        items = [r for r in items if r.get("subset") == subset_filter]

    n_samples = cfg.get("n_samples")
    if n_samples is not None:
        items = items[: int(n_samples)]

    print(f"RACE eval: {len(items)} problems from {test_path}")
    print(f"  subset_filter={subset_filter}")
    print(f"  Each problem -> 4 score_continuations calls (" + " ".join(LETTERS) + ")")

    correct = 0
    examples: list[dict] = []
    t0 = time.time()
    options = [f" {L}" for L in LETTERS]

    for i, rec in enumerate(items):
        # RACE records carry the FULL pre-rendered question (already including
        # "Multiple Choice question:", passage, "- ... =A" choices, and the
        # "Respond only with..." instruction). Wrap into our chat prompt shape.
        prompt = f"User: {rec['question'].strip()}\n\nAssistant:"
        scores = backend.score_continuations(prompt, options)
        pred_idx = max(range(len(scores)), key=lambda j: scores[j])
        pred_letter = LETTERS[pred_idx]
        gold_letter = str(rec["expected"]).strip().upper()
        is_correct = pred_letter == gold_letter
        correct += int(is_correct)

        examples.append({
            "subset": rec.get("subset"),
            "question_excerpt": rec["question"][:160] + "...",
            "gold": gold_letter,
            "pred": pred_letter,
            "logprobs": dict(zip(LETTERS, scores)),
            "correct": is_correct,
        })

        if (i + 1) % 50 == 0 or i == len(items) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(items)}] correct={correct}  "
                  f"acc={correct/(i+1):.3f}  ({rate:.2f} probs/s)")

    accuracy = correct / len(items) if items else 0.0
    # Also report per-subset accuracy if both subsets are present
    by_subset: dict[str, dict] = {}
    for ex in examples:
        s = ex.get("subset") or "unknown"
        if s not in by_subset:
            by_subset[s] = {"correct": 0, "n": 0}
        by_subset[s]["n"] += 1
        by_subset[s]["correct"] += int(ex["correct"])
    for s, agg in by_subset.items():
        agg["accuracy"] = agg["correct"] / agg["n"] if agg["n"] else 0.0

    return {
        "metric": "accuracy",
        "pass_at_1": accuracy,
        "accuracy": accuracy,
        "n_correct": correct,
        "n_total": len(items),
        "by_subset": by_subset,
        "test_set": str(test_path),
        "examples": examples,
    }
