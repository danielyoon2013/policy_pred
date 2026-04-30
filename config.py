"""Config for policy_pred: per-year midtrain/SFT pipeline + belief elicitation."""
from pathlib import Path

DATA_ROOT = Path("D:/hist_LLM/policy_pred")
CORPUS_RAW_ROOT = Path("D:/hist_LLM/corpus/raw")

# Base model is assumed trained on data <= BASE_CUTOFF_YEAR.
# Year-models are built cumulatively: base -> +1931 -> +1932 -> ...
BASE_CUTOFF_YEAR = 1930
START_YEAR = 1931
END_YEAR = 1970

# Starting model for cumulative year-by-year training. References a name from
# policy_pred/models/registry.yaml; flip to 'nanochat_1925' (or any other entry)
# to compare backends without touching the pipeline.
BASE_MODEL = "talkie_base"


def base_model_spec() -> dict:
    """Resolve BASE_MODEL to a backend spec via the model registry."""
    from .models.registry import resolve
    return resolve(BASE_MODEL)

# Synthetic data generation.
OPENAI_MODEL = "gpt-4o-mini"
ITEMS_PER_YEAR = 5000

# Belief elicitation.
PROBES_PER_POLICY = 16   # paraphrases per (policy, year-model)
PROBE_OPTIONS = 4        # MC arity: 1 correct + 3 plausible distractors
POLICY_CATALOG_PATH = Path(__file__).parent / "policies" / "catalog.yaml"


def year_dir(year: int) -> Path:
    return DATA_ROOT / "years" / str(year)


def year_corpus_path(year: int) -> Path:
    return year_dir(year) / "raw.parquet"


def year_sft_path(year: int) -> Path:
    return year_dir(year) / "sft.jsonl"


def year_checkpoint_dir(year: int, stage: str) -> Path:
    # stage in {"midtrain", "sft"}
    return year_dir(year) / "checkpoint" / stage


def policy_eval_path(year: int, policy_id: str) -> Path:
    return DATA_ROOT / "eval" / policy_id / f"{year}.json"
