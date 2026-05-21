"""
The MathQA dataset.
https://huggingface.co/datasets/allenai/math_qa

Multiple-choice math word problems (5 choices: a-e).
Tests quantitative reasoning: arithmetic, rates, geometry, probability.
"""

import re
from datasets import load_dataset
from tasks.common import Task, render_mc

class MathQA(Task):

    letters = ("A", "B", "C", "D", "E")

    def __init__(self, split, cache_dir=None, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "validation", "test"], "MathQA split must be train|validation|test"
        # allenai/math_qa uses a legacy loading script that newer versions of
        # the datasets library no longer support. Load from the auto-converted
        # parquet files on the HF Hub instead.
        try:
            self.ds = load_dataset("allenai/math_qa", split=split, cache_dir=cache_dir).shuffle(seed=42)
        except Exception:
            # Legacy loading script no longer supported in datasets >= 3.0.
            # Load from auto-converted parquet on the refs/convert/parquet branch.
            self.ds = load_dataset(
                "parquet",
                data_files=f"hf://datasets/allenai/math_qa@~parquet/default/{split}/0000.parquet",
                split="train",
                cache_dir=cache_dir,
            ).shuffle(seed=42)

    @staticmethod
    def parse_options(options_str):
        """Parse option string like 'a ) 24 , b ) 120 , c ) 625 , d ) 720 , e ) 1024'"""
        choices = re.findall(r'[a-e] \) (.+?)(?= , [a-e] \)|$)', options_str)
        assert len(choices) == 5, f"Expected 5 choices, got {len(choices)} from: {options_str}"
        return [c.strip() for c in choices]

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["Problem"]
        choices = self.parse_options(row["options"])
        answer_string = row["correct"].upper()  # "c" -> "C"
        assert answer_string in self.letters, f"MathQA answer {answer_string} must be one of {self.letters}"
        user_message = render_mc(question, self.letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string}
        ]
        return {"messages": messages, "letters": self.letters}

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in self.letters
        return assistant_response == conversation['messages'][-1]['content']
