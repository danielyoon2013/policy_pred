"""
The PIQA (Physical Interaction QA) dataset.
https://huggingface.co/datasets/ybisk/piqa

Physical commonsense reasoning: given a goal, pick which of two solutions
is more physically plausible. Time-invariant: physics doesn't change.
"""

from datasets import load_dataset
from tasks.common import Task, render_mc

class PIQA(Task):

    letters = ("K", "V")

    def __init__(self, split, cache_dir=None, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "validation"], "PIQA split must be train|validation"
        try:
            self.ds = load_dataset("ybisk/piqa", split=split, cache_dir=cache_dir).shuffle(seed=42)
        except Exception:
            # Fallback for datasets versions that removed script-based loading.
            # Load the raw JSONL + labels directly from HuggingFace Hub.
            self.ds = self._load_from_hub(split, cache_dir).shuffle(seed=42)

    @staticmethod
    def _load_from_hub(split, cache_dir=None):
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        from datasets import Dataset
        # Download the auto-converted parquet file from HuggingFace's convert branch
        parquet_path = hf_hub_download(
            "ybisk/piqa",
            f"plain_text/{split}/0000.parquet",
            repo_type="dataset",
            revision="refs/convert/parquet",
            cache_dir=cache_dir,
        )
        table = pq.read_table(parquet_path)
        return Dataset(table)

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["goal"]
        choices = [row["sol1"], row["sol2"]]
        answer_idx = row["label"]  # 0 or 1
        answer_string = self.letters[answer_idx]
        # create and return the Conversation object
        user_message = render_mc(question, self.letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string}
        ]
        conversation = {
            "messages": messages,
            "letters": self.letters,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in self.letters, f"PIQA answer {assistant_response} is expected to be one of {self.letters}"
        assistant_message = conversation['messages'][-1]['content']
        return assistant_response == assistant_message
