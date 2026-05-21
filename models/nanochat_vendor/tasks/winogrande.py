"""
The Winogrande dataset.
https://huggingface.co/datasets/allenai/winogrande

Coreference resolution task: given a sentence with a blank (_),
choose which of two options correctly fills the blank.
Time-invariant: tests nuanced language understanding, not factual knowledge.
"""

from datasets import load_dataset
from tasks.common import Task, render_mc

class Winogrande(Task):

    letters = ("K", "V")

    def __init__(self, split, cache_dir=None, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "validation"], "Winogrande split must be train|validation"
        self.ds = load_dataset("allenai/winogrande", "winogrande_debiased", split=split, cache_dir=cache_dir).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        sentence = row["sentence"]  # contains _ as the blank
        choices = [row["option1"], row["option2"]]
        answer_idx = int(row["answer"]) - 1  # "1" or "2" -> 0 or 1
        answer_string = self.letters[answer_idx]
        # create and return the Conversation object
        question = f"Fill in the blank: {sentence}"
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
        assert assistant_response in self.letters, f"Winogrande answer {assistant_response} is expected to be one of {self.letters}"
        assistant_message = conversation['messages'][-1]['content']
        return assistant_response == assistant_message
