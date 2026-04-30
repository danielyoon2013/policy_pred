"""Build per-year text shards used as the seed for synthetic generation and midtrain.

Reads from CORPUS_RAW_ROOT (existing pipeline's per-year parquets) and writes a
filtered, deduped year shard to year_corpus_path(year). Likely steps:
  1. Load D:/hist_LLM/corpus/raw/{year}.parquet
  2. Apply quality filter (existing Ridge classifier per period)
  3. Optional dedupe vs. base-model training data
  4. Write year_corpus_path(year)
"""


def build(year: int) -> None:
    raise NotImplementedError
