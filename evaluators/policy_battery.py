"""Policy belief-elicitation evaluator with yes/no + Likert scoring.

Reads policy descriptions from EITHER `evaluators/catalog.yaml` (the V1
5-policy paraphrased catalog) OR `us_policy_event_battery_v4.csv` (the
211-event battery). Filter to a subset via `cfg.filter`.

Two scoring modes (run both in one call):

  yes_no
    Prompt: "Is it likely that the United States has [description]?\\n\\nAnswer:"
    Score:  [" Yes", " No"]
    Output: P(yes) = exp(logp_Yes) / (exp(logp_Yes) + exp(logp_No))

  likert5
    Prompt: "How likely is it that the United States has [description]?\\n\\nAnswer:"
    Score:  [" Very likely", " Likely", " Uncertain", " Unlikely", " Very unlikely"]
    Output: weighted score, weights [1.0, 0.75, 0.5, 0.25, 0.0]

Critical design decisions:
  - No year token in the prompt. Year-Y model's answer reflects its IMPLICIT
    world-model anchored by training data, not a year-token in the prompt.
  - Description is a paraphrase. Never include the policy's canonical name
    ("Social Security Act" → "national program providing income to retired
    workers, funded by mandatory payroll contributions").
  - Both modes run in a single eval.py invocation; output goes into one
    eval.json under results.policies[].scores.{yes_no, likert5}.

Config keys (from experiment YAML eval section):
  source: "catalog" | "csv"           (default: "catalog")
  catalog_path: <path>                 (used if source="catalog")
  csv_path: <path>                     (used if source="csv")
  filter:
    id: [list of ids to keep]          (filter by event_id or catalog id)
  modes: ["yes_no", "likert5"]         (default: both)
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any


EVALUATOR_VERSION = "v4.0"  # v4: centered likert (-1..+1) + agreement options + normative-form prompts

# Agreement-form likert (works naturally for normative "Should X?" questions). Centered
# so Uncertain = 0, Strongly agree = +1, Strongly disagree = -1. P(yes_no) maps the same
# semantic axis to a 0..1 scale; the two modes are complementary uncertainty estimates.
LIKERT_OPTIONS = [" Strongly agree", " Agree", " Uncertain", " Disagree", " Strongly disagree"]
LIKERT_VALUES = [1.0, 0.5, 0.0, -0.5, -1.0]

YESNO_OPTIONS = [" Yes", " No"]


# ---------------------------------------------------------------------------
# Loading policies (catalog OR CSV)
# ---------------------------------------------------------------------------

def _load_catalog(path: Path) -> list[dict]:
    """Load the V1 5-policy YAML catalog. Returns list of normalized dicts.

    The catalog stores `description` as a factual paraphrase (e.g. "national
    program providing income to retired workers..."). We synthesize a
    normative question from it ("Should the United States have X?") so the
    catalog path uses the same probe shape as the CSV path.
    """
    import yaml
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    out: list[dict] = []
    for p in data["policies"]:
        description = " ".join(p["description"].split())
        out.append({
            "id": p["id"],
            "question": f"Should the United States have {description}?",
            "description": description,
            "implementation_year": p.get("implementation_year"),
            "anticipation_start_year": p.get("anticipation_start_year"),
        })
    return out


def _load_csv(path: Path) -> list[dict]:
    """Load us_policy_event_battery_v4.csv. Returns list of normalized dicts.

    Pulls `benchmark_question` (already in normative "Should X?" form) directly
    from the CSV — that's what the probe scores. Keeps `adoption_object` as a
    `description` field for display + plot labels.
    """
    import pandas as pd
    df = pd.read_csv(path)
    out: list[dict] = []
    for _, row in df.iterrows():
        question = str(row.get("benchmark_question", "")).strip()
        if not question:
            # Fallback: synthesize from adoption_object if benchmark_question is empty.
            description = " ".join(str(row["adoption_object"]).split())
            question = f"Should the United States have {description}?"
        out.append({
            "id": str(row["event_id"]),
            "question": question,
            "description": " ".join(str(row["adoption_object"]).split()),
            "implementation_year": int(row["event_year"]),
            "anticipation_start_year": None,  # not in CSV
            "domain": str(row.get("domain", "")),
            "institution": str(row.get("institution", "")),
        })
    return out


def _apply_filter(items: list[dict], filter_cfg: dict) -> list[dict]:
    """Apply cfg.filter to the policy list."""
    if not filter_cfg:
        return items
    keep_ids = filter_cfg.get("id")
    if keep_ids:
        keep_set = set(keep_ids)
        items = [i for i in items if i["id"] in keep_set]
    return items


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _build_yesno_prompt(question: str) -> str:
    """Yes/no prompt. The question is a normative 'Should X?' phrasing."""
    return f"{question}\n\nAnswer:"


def _build_likert_prompt(question: str) -> str:
    """Agreement-style likert prompt. The question is a normative 'Should X?' phrasing."""
    return (
        f"Respond to the following question by rating your agreement:\n"
        f"\n"
        f"{question}\n"
        f"\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------------
# Shared chat / chain-of-thought (CoT) prompt builders
#
# These are the SINGLE SOURCE OF TRUTH used by BOTH the SFT generator
# (synth/sft_corpus) and the CoT eval, so a CoT-SFT record and the CoT eval
# prompt are byte-identical up to the answer position. They reproduce
# train.py:render_chat_record's rendering for one user->assistant turn:
#   "User: {user}\n\nAssistant: {assistant}"  (content .strip()'d)
# the prompt side being everything up to and including "\n\nAssistant:".
# ---------------------------------------------------------------------------

ANSWER_MARKER = "Answer:"
COT_CUE_YESNO = "Think step by step about the reasons, then answer Yes or No."
COT_CUE_LIKERT = "Think step by step about the reasons, then give your rating."


def chat_prompt(user_content: str) -> str:
    """Prompt side of a single user->assistant chat turn (matches render_chat_record).

    Used when evaluating an SFT-trained model so the eval context reproduces the
    exact string the model was trained to continue. `chat_format=False` callers
    use the bare builders instead (for the untrained-on-chat LoRA-only baseline).
    """
    return f"User: {user_content.strip()}\n\nAssistant:"


def build_yesno_cot_user(question: str) -> str:
    """User-turn content for the Yes/No CoT probe (reason, then answer)."""
    return f"{question}\n\n{COT_CUE_YESNO}"


def build_likert_cot_user(question: str) -> str:
    """User-turn content for the Likert CoT probe (reason, then rate)."""
    return (f"Respond to the following question by rating your agreement.\n"
            f"\n{question}\n\n{COT_CUE_LIKERT}")


def cot_assistant(reasoning: str, label: str) -> str:
    """Assistant turn for a CoT-SFT record: reasoning, then the verdict after
    the answer marker. Mirrors what the CoT eval generates + scores."""
    return f"{reasoning.strip()}\n\n{ANSWER_MARKER} {label.strip()}"


def _score_yesno(backend, question: str) -> dict:
    """Score the yes/no probe and return P(yes) + raw logprobs."""
    prompt = _build_yesno_prompt(question)
    scores = backend.score_continuations(prompt, YESNO_OPTIONS)
    logp_yes, logp_no = scores
    # Normalize via log-sum-exp for stability.
    m = max(logp_yes, logp_no)
    p_yes = math.exp(logp_yes - m)
    p_no = math.exp(logp_no - m)
    total = p_yes + p_no
    return {
        "prompt": prompt,
        "logprobs": {"Yes": logp_yes, "No": logp_no},
        "p_yes": p_yes / total,
    }


def _score_likert(backend, question: str) -> dict:
    """Score the 5-option agreement likert; return weighted (centered) score + probs."""
    prompt = _build_likert_prompt(question)
    scores = backend.score_continuations(prompt, LIKERT_OPTIONS)
    # Softmax over the 5 options for a proper distribution.
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    probs = [e / total for e in exps]
    weighted = sum(p * v for p, v in zip(probs, LIKERT_VALUES))
    return {
        "prompt": prompt,
        "logprobs": dict(zip([o.strip() for o in LIKERT_OPTIONS], scores)),
        "probs": dict(zip([o.strip() for o in LIKERT_OPTIONS], probs)),
        "score": weighted,  # centered: [-1, +1]; positive = agree
    }


# ---------------------------------------------------------------------------
# Public entry point (called by eval.py dispatcher)
# ---------------------------------------------------------------------------

def run(backend, cfg: dict) -> dict:
    """Run the policy battery against a model. Returns a dict for eval.json."""
    source = cfg.get("source", "catalog")
    if source == "catalog":
        path = Path(cfg.get("catalog_path") or
                    Path(__file__).parent / "catalog.yaml")
        policies = _load_catalog(path)
    elif source == "csv":
        path = Path(cfg.get("csv_path") or "us_policy_event_battery_v4.csv")
        policies = _load_csv(path)
    else:
        raise ValueError(f"unknown source: {source!r} (expected 'catalog' or 'csv')")

    print(f"  loaded {len(policies)} policies from {path}")

    policies = _apply_filter(policies, cfg.get("filter") or {})
    if not policies:
        raise ValueError(f"no policies survived filter: {cfg.get('filter')}")
    print(f"  after filter: {len(policies)} policies")

    modes = cfg.get("modes") or ["yes_no", "likert5"]
    print(f"  modes: {modes}")

    results: list[dict] = []
    for policy in policies:
        t0 = time.time()
        scores: dict[str, Any] = {}

        if "yes_no" in modes:
            scores["yes_no"] = _score_yesno(backend, policy["question"])

        if "likert5" in modes:
            scores["likert5"] = _score_likert(backend, policy["question"])

        # One-line summary per policy for visual scan.
        summary = f"  {policy['id']:<32s}"
        if "yes_no" in scores:
            summary += f"  P(yes)={scores['yes_no']['p_yes']:.3f}"
        if "likert5" in scores:
            summary += f"  likert={scores['likert5']['score']:+.3f}"  # centered: show sign
        print(summary)

        results.append({
            "id": policy["id"],
            "implementation_year": policy.get("implementation_year"),
            "anticipation_start_year": policy.get("anticipation_start_year"),
            "question": policy["question"],
            "description": policy.get("description"),
            "scores": scores,
            "elapsed_sec": time.time() - t0,
        })

    return {
        "evaluator": "policy_battery",
        "evaluator_version": EVALUATOR_VERSION,
        "source": source,
        "n_policies": len(policies),
        "modes": modes,
        "policies": results,
    }
