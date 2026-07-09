"""Bridge evaluator: score any vendored nanochat `Task` via a policy_pred backend.

One evaluator that brings the whole nanochat benchmark suite into this harness —
the LAB look-ahead test (test 3) plus external capability benchmarks (test 4:
HellaSwag / Winogrande / PIQA / ARC / MMLU / MMLU-History) — without a file per
benchmark. It reuses each nanochat Task's own question rendering (render_mc) and
correctness check (Task.evaluate), and scores multiple-choice by length-normalized
log-prob over the answer letters (backend.score_continuations), exactly like the
native benchmark/arc.py evaluator.

Config (experiment YAML `eval:` block):
    evaluator: nanochat_task
    task: lab            # one of _TASKS below (+ the "mmlu_history" alias)
    # local-file tasks (lab, internal_mc):
    data_file: D:/hist_LLM/eval_data/lab_eval.jsonl
    # HF tasks (hellaswag, winogrande, piqa, arc, mmlu[_history]):
    subset: challenge    # required for arc/mmlu; ignored otherwise
    split: validation
    cache_dir: D:/hist_LLM/eval_data/hf_cache
    # optional:
    n_samples: 500       # cap the number of items (default: all)
    subjects: [...]      # override the MMLU-History subject filter
    prompt_template: "User: {q}\n\nAssistant:"   # how the MC prompt is framed

Returns the same shape as benchmark/arc.py:
    {metric, accuracy, pass_at_1, n_correct, n_total, task, examples[]}

Parity note: scoring uses the raw render_mc prompt wrapped in User:/Assistant:
and " <letter>" continuations (policy_pred's convention, matching arc_mc), NOT
nanochat chat_eval's chat-templated last-position logits. Numbers are internally
consistent across years/tasks in this harness; they are not guaranteed identical
to a nanochat chat_eval run.
"""
from __future__ import annotations

import sys
import time
from importlib import import_module
from pathlib import Path

# Make the vendored nanochat `tasks` package importable (tasks/*.py do
# `from tasks.common import ...`). Vendor lives at <repo>/models/nanochat_vendor/.
_VENDOR = Path(__file__).resolve().parents[2] / "models" / "nanochat_vendor"
if _VENDOR.exists() and str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

# public task name -> (vendor module, class, kind). kind "file" needs cfg.data_file;
# "hf" needs cfg.split (+ cfg.subset for arc/mmlu).
_TASKS = {
    "lab":         ("tasks.lab_eval",    "LABEval",    "file"),
    "internal_mc": ("tasks.internal_mc", "InternalMC", "file"),
    "hellaswag":   ("tasks.hellaswag",   "HellaSwag",  "hf"),
    "winogrande":  ("tasks.winogrande",  "Winogrande", "hf"),
    "piqa":        ("tasks.piqa",        "PIQA",       "hf"),
    "arc":         ("tasks.arc",         "ARC",        "hf"),
    "mmlu":        ("tasks.mmlu",        "MMLU",       "hf"),
}

# MMLU subjects that count as "history" for the MMLU-History slice.
_HISTORY_SUBJECTS = (
    "high_school_us_history",
    "high_school_world_history",
    "high_school_european_history",
    "prehistory",
)


def _build_task(cfg: dict):
    """Instantiate the requested nanochat Task from cfg."""
    name = cfg["task"]

    # MMLU-History: full MMLU filtered to history subjects (subjects already exist
    # in cais/mmlu; nanochat's MMLU stores `subject` per row).
    if name == "mmlu_history":
        MMLU = getattr(import_module("tasks.mmlu"), "MMLU")
        task = MMLU(subset=cfg.get("subset", "all"), split=cfg.get("split", "test"))
        subjects = set(cfg.get("subjects") or _HISTORY_SUBJECTS)
        task.ds = task.ds.filter(lambda r: r["subject"] in subjects)
        return task

    if name not in _TASKS:
        raise KeyError(f"unknown nanochat task {name!r}; known: {sorted(list(_TASKS) + ['mmlu_history'])}")
    module, cls_name, kind = _TASKS[name]
    cls = getattr(import_module(module), cls_name)

    if kind == "file":
        data_file = cfg.get("data_file")
        if not data_file:
            raise ValueError(f"task {name!r} needs cfg.data_file")
        return cls(data_file)

    # hf task
    kwargs = {"split": cfg.get("split", "validation")}
    if cfg.get("subset") is not None:
        kwargs["subset"] = cfg["subset"]     # required by arc/mmlu, ignored by others
    if cfg.get("cache_dir"):
        kwargs["cache_dir"] = cfg["cache_dir"]
    return cls(**kwargs)


def run(backend, cfg: dict) -> dict:
    task = _build_task(cfg)
    name = cfg["task"]
    n = len(task)
    n_samples = cfg.get("n_samples")
    total = n if n_samples is None else min(n, int(n_samples))
    template = cfg.get("prompt_template", "User: {q}\n\nAssistant:")
    categorical = task.eval_type == "categorical"
    max_new = int(cfg.get("max_new_tokens", 256))

    print(f"nanochat_task '{name}' ({task.eval_type}): {total}/{n} items")

    correct = 0
    examples: list[dict] = []
    t0 = time.time()
    for i in range(total):
        conv = task[i]
        prompt = template.format(q=conv["messages"][0]["content"])
        gold = conv["messages"][-1]["content"]
        if categorical:
            letters = list(conv["letters"])
            scores = backend.score_continuations(prompt, [f" {L}" for L in letters])
            pred = letters[max(range(len(scores)), key=lambda j: scores[j])]
        else:
            pred = backend.generate(prompt, max_new_tokens=max_new)
        ok = bool(task.evaluate(conv, pred))
        correct += int(ok)
        examples.append({"index": i, "gold": gold, "pred": pred, "correct": ok})

        if (i + 1) % 50 == 0 or i == total - 1:
            rate = (i + 1) / max(time.time() - t0, 1e-9)
            print(f"  [{i+1}/{total}] correct={correct} acc={correct/(i+1):.3f} ({rate:.1f} it/s)")

    accuracy = correct / total if total else 0.0
    return {
        "metric": "accuracy",
        "task": name,
        "pass_at_1": accuracy,
        "accuracy": accuracy,
        "n_correct": correct,
        "n_total": total,
        "examples": examples,
    }
