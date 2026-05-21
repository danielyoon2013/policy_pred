"""
The HellaSwag dataset.
https://huggingface.co/datasets/Rowan/hellaswag

Sentence completion task: given a context, pick the most plausible continuation
from 4 choices. Tests language understanding and commonsense reasoning.
Time-invariant: scenarios are generic everyday activities.
"""

from datasets import load_dataset
from tasks.common import Task, render_mc

class HellaSwag(Task):

    letters = ("Q", "X", "Z", "J")

    def __init__(self, split, cache_dir=None, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "validation"], "HellaSwag split must be train|validation"
        self.ds = load_dataset("Rowan/hellaswag", split=split, cache_dir=cache_dir).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["ctx"]  # the context/premise
        choices = row["endings"]  # list of 4 possible continuations
        answer_idx = int(row["label"])  # "0"-"3" as string -> int
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
        assert assistant_response in self.letters, f"HellaSwag answer {assistant_response} is expected to be one of {self.letters}"
        assistant_message = conversation['messages'][-1]['content']
        return assistant_response == assistant_message
