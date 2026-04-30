"""OpenAI batch API -> per-year synthetic SFT data.

Reads year_corpus_path(year), builds prompts via prompts.WORLDVIEW_TEMPLATES, submits
to OpenAI batch, writes year_sft_path(year) as JSONL of {messages: [...]} records.
Reuse src/post_training/utils.py helpers (write_jsonl, validate_conversation) where
they fit.
"""


def generate_year(year: int) -> None:
    raise NotImplementedError
