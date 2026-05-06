"""GSM8K open-ended math evaluator.

Loads GSM8K test set, runs greedy generation per problem, parses the final
"#### N" answer, and reports pass@1 (% of problems where the parsed answer
matches the ground truth integer).

CPU works but is slow (no KV-cache). Run on remote A100 for production
eval; budget ~1-3 hours per model on 1.3K problems at 256 generated tokens.

Config keys (from experiment YAML eval section):
    test_set: str         HF dataset spec (default "gsm8k:main:test")
    n_samples: int        cap test items (default: all ~1319)
    max_new_tokens: int   generation length (default 256)
    parser: str           "regex_final_number" (default; the `#### N` format)
"""
from __future__ import annotations

import re
import time
from typing import Optional


# Standard GSM8K few-shot prompt prefix (from the original GSM8K paper).
# Five examples, each ending in "#### N", to teach the model the format.
GSM8K_FEW_SHOT_PROMPT = """Solve each problem step by step, then state the final answer after `####`.

Q: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
A: Natalia sold 48/2 = 24 clips in May.
Natalia sold 48+24 = 72 clips altogether in April and May.
#### 72

Q: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
A: Weng earns 12/60 = $0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $10.
#### 10

Q: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
A: Betty has 100/2 = $50.
Betty's grandparents gave her 15*2 = $30.
This means Betty has 50+15+30 = $95.
Betty needs 100-95 = $5 more.
#### 5

Q: {question}
A:"""


def _extract_answer(text: str) -> Optional[str]:
    """Pull the final '#### N' answer from generated text.

    Returns the answer as a normalized string (e.g. '72', '0.5', '-3').
    Returns None if no parseable answer is found.
    """
    # Look for `#### <number>` (last occurrence, in case of multiple).
    matches = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "").strip()
    # Normalize: strip trailing zeros after decimal.
    try:
        f = float(raw)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        return None


def _normalize_gold(gold: str) -> str:
    """GSM8K gold answers come as '... #### 72'. Extract the number."""
    return _extract_answer(gold) or gold.strip()


def run(backend, cfg: dict) -> dict:
    """Run GSM8K eval against a TalkieBackend instance.

    `backend` is a TalkieBackend (or PEFT-wrapped equivalent) with the
    adapter already applied. Returns a dict ready to JSON-dump.
    """
    from datasets import load_dataset

    spec = cfg.get("test_set", "gsm8k:main:test")
    name, subset, split = spec.split(":") if spec.count(":") == 2 else ("gsm8k", "main", "test")
    print(f"Loading dataset {name}/{subset}/{split}...")
    ds = load_dataset(name, subset, split=split)

    n_samples = cfg.get("n_samples")
    if n_samples is not None:
        ds = ds.select(range(min(n_samples, len(ds))))

    max_new_tokens = cfg.get("max_new_tokens", 256)
    print(f"Evaluating on {len(ds):,} problems "
          f"(max_new_tokens={max_new_tokens})...")

    correct = 0
    examples: list[dict] = []
    t0 = time.time()
    for i, item in enumerate(ds):
        question = item["question"]
        gold_text = item["answer"]
        gold = _normalize_gold(gold_text)

        prompt = GSM8K_FEW_SHOT_PROMPT.format(question=question)
        gen = backend.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_sequences=["\nQ:", "\n\nQ:"],
        )
        pred = _extract_answer(gen)
        is_correct = pred is not None and pred == gold
        correct += int(is_correct)

        examples.append({
            "question": question,
            "gold": gold,
            "pred": pred,
            "generation": gen,
            "correct": is_correct,
        })

        if (i + 1) % 25 == 0 or i == len(ds) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(ds)}] correct={correct}  "
                  f"pass@1={correct/(i+1):.3f}  ({rate:.2f} probs/s)")

    pass_at_1 = correct / len(ds) if len(ds) else 0.0
    return {
        "metric": "pass@1",
        "pass_at_1": pass_at_1,
        "n_correct": correct,
        "n_total": len(ds),
        "test_set": spec,
        "max_new_tokens": max_new_tokens,
        "examples": examples,
    }
