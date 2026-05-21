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
