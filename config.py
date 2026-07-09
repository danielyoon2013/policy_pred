"""Config for policy_pred: paths, year range, hyperparameters."""
import os
from pathlib import Path

# DATA_ROOT can be overridden via env var so the same code runs on the local
# Windows workstation (D:/...) and rented Linux GPU boxes (~/policy_pred_data).
DATA_ROOT = Path(os.environ.get("POLICY_PRED_DATA_ROOT")
                 or "D:/hist_LLM/policy_pred")
CORPUS_RAW_ROOT = Path("D:/hist_LLM/corpus/raw")

# TALKIE_WEIGHTS_DIR can be overridden via env var so the same code runs on
# the local Windows workstation (D:/...) and rented Linux GPU boxes ($HOME/...).
TALKIE_WEIGHTS_DIR = Path(
    os.environ.get("TALKIE_WEIGHTS_DIR") or DATA_ROOT / "models" / "talkie_base"
)

# Nanochat-1900-1949 base model. The default points at the trained
# checkpoint in the hist_LLM tree on the local Windows box; override
# via NANOCHAT_WEIGHTS_DIR on remote pods. The dir must contain
# tokenizer/ + base_checkpoints/d<depth>/ (nanochat's convention).
NANOCHAT_WEIGHTS_DIR = Path(
    os.environ.get("NANOCHAT_WEIGHTS_DIR")
    or "D:/hist_LLM/periods/1900_1949/model"
)
QWEN_WEIGHTS_DIR = Path(
    os.environ.get("QWEN_WEIGHTS_DIR") or DATA_ROOT / "models" / "qwen_base"
)

# Continual per-year "vintage" nanochat base checkpoints (one per year-boundary,
# written by the base-training run as continual/model/year_checkpoints/d<depth>/
# model_<step>.pt). The year->step map (below) lets an experiment load a SPECIFIC
# year's checkpoint directly, without staging one dir per year. Env-overridable so
# the same code runs on Windows (D:) and on the cluster (/project/bybeedata).
CONTINUAL_WEIGHTS_DIR = Path(
    os.environ.get("CONTINUAL_WEIGHTS_DIR") or "D:/hist_LLM/continual/model"
)
# Dir that CONTAINS tokenizer/ for the continual run. nanochat's get_tokenizer()
# reads NANOCHAT_BASE_DIR/tokenizer/, which for the continual checkpoints lives in
# a sibling dir (continual/data), not inside the checkpoint dir.
CONTINUAL_TOKENIZER_DIR = Path(
    os.environ.get("CONTINUAL_TOKENIZER_DIR") or "D:/hist_LLM/continual/data"
)

# Base model is assumed trained on data <= BASE_CUTOFF_YEAR.
# Year-models are built cumulatively: base -> +1931 -> +1932 -> ...
BASE_CUTOFF_YEAR = 1930
START_YEAR = 1931
END_YEAR = 1970

# Belief elicitation.
PROBES_PER_POLICY = 16   # paraphrases per (policy, year-model)
PROBE_OPTIONS = 4        # MC arity: 1 correct + 3 plausible distractors
POLICY_CATALOG_PATH = Path(__file__).parent / "policies" / "catalog.yaml"

# V2: synthetic data generation (deferred, not used in V1).
# OPENAI_MODEL = "gpt-4o-mini"
# ITEMS_PER_YEAR = 5000


def year_dir(year: int) -> Path:
    return DATA_ROOT / "years" / str(year)


def year_corpus_path(year: int) -> Path:
    return year_dir(year) / "raw.parquet"


def year_checkpoint_dir(year: int) -> Path:
    return year_dir(year) / "checkpoint"


def policy_eval_path(year: int, policy_id: str) -> Path:
    return DATA_ROOT / "eval" / policy_id / f"{year}.json"


# year -> continual training step, for loading a specific year's vintage base.
# Searched in order: $YEAR_STEP_MAP, the all-year (90) map, then the 23-year subset.
_YEAR_STEP_MAPS = [
    p for p in (
        Path(os.environ["YEAR_STEP_MAP"]) if os.environ.get("YEAR_STEP_MAP") else None,
        Path(__file__).parent / "docs" / "findings" / "2026-06"
        / "continuous_d23_all_year_rel20" / "year_step_map_all.csv",
        Path(__file__).parent / "scripts" / "pod" / "continuous_d23_rel20" / "year_step_map.csv",
    ) if p is not None
]


def year_to_step(year: int, model_tag: str = "d23") -> int:
    """Look up the continual checkpoint step for a model-year (from the year_step maps)."""
    import csv
    for path in _YEAR_STEP_MAPS:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["year"]) == int(year):
                    return int(row["step"])
    raise KeyError(f"year {year} not found in any year_step map: {_YEAR_STEP_MAPS}")
