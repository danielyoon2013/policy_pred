"""Values-track evaluator: probe a vintage model with real survey questions.

The model is treated as a survey respondent of its training era (the Kaushik/Leland
design decision): each battery item's near-original question is posed bare, and the
survey's own answer options are scored as continuations. The tracked quantity is
p_focal — the probability mass the model puts on the item's focal answer(s) — which
downstream analysis compares, vintage-by-vintage, against the real GSS/Gallup
focal shares in data_artifacts/values_truth_v1.csv.

Deliberately NO chain-of-thought mode (reasoning inflates P(yes) on models never
trained to reason — see the reasoning-gate finding) and NO year-window filtering
(every vintage answers every item; the cross-vintage trend is the signal).

Config keys (YAML eval: block):
  csv_path      battery CSV (default us_values_battery_v1.csv at repo root)
  chat_format   wrap prompts in User:/Assistant: rendering (default False — the
                LoRA-only vintages are probed bare, matching the policy track)
"""
from __future__ import annotations

import csv
import math
import re
import time
from pathlib import Path

try:
    from ..policy.battery import chat_prompt  # noqa: TID252
except ImportError:  # standalone import for local testing
    from policy.battery import chat_prompt  # type: ignore[no-redef]

EVALUATOR_VERSION = "values-v1.0"


def _load_battery(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            options = [o.strip() for o in row["answer_options"].split("|") if o.strip()]
            focal = [o.strip() for o in row["focal_answer"].split(" + ") if o.strip()]
            missing = [o for o in focal if o not in options]
            if missing:
                raise ValueError(f"{row['item_id']}: focal answers not in options: {missing}")
            items.append(dict(row, options=options, focal=focal))
    if not items:
        raise ValueError(f"empty battery at {path}")
    return items


def _softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def run(backend, cfg: dict) -> dict:
    """Score every battery item; return per-item p_focal + full option distributions."""
    csv_path = Path(cfg.get("csv_path") or "us_values_battery_v1.csv")
    chat_format = bool(cfg.get("chat_format", False))
    items = _load_battery(csv_path)

    exp_name = cfg.get("_experiment_name", "")
    m = re.search(r"_(\d{4})_", exp_name) or re.search(r"(\d{4})", exp_name)
    model_year = int(m.group(1)) if m else None

    print(f"  values battery: {len(items)} items from {csv_path}  "
          f"(model_year={model_year}, chat_format={chat_format})")

    results = []
    for it in items:
        t0 = time.time()
        prompt = f"{it['question_text']}\n\nAnswer:"
        if chat_format:
            prompt = chat_prompt(prompt)
        scores = backend.score_continuations(prompt, [" " + o for o in it["options"]])
        probs = _softmax(list(scores))
        p_per_option = dict(zip(it["options"], probs))
        p_focal = sum(p_per_option[o] for o in it["focal"])
        results.append({
            "item_id": it["item_id"],
            "question_uid": it["question_uid"],
            "focal_answer": it["focal_answer"],
            "p_focal": p_focal,
            "p_per_option": p_per_option,
            "value_construct": it["value_construct"],
            "paired_event_ids": it["paired_event_ids"],
            "elapsed_sec": time.time() - t0,
        })
        print(f"  {it['item_id']:<18} p_focal={p_focal:.3f}  "
              f"({' '.join(f'{o[:12]}={p:.2f}' for o, p in p_per_option.items())})")

    return {
        "evaluator": "values_battery",
        "evaluator_version": EVALUATOR_VERSION,
        "csv_path": str(csv_path),
        "model_year": model_year,
        "chat_format": chat_format,
        "n_items": len(results),
        "items": results,
    }
