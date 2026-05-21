"""
The BoolQ dataset.
https://huggingface.co/datasets/google/boolq

Reading comprehension task: given a passage and a question, answer Yes or No.
Time-invariant: the answer is always derivable from the provided passage,
so no external knowledge of any time period is required.
"""

from datasets import load_dataset
from tasks.common import Task, render_mc

class BoolQ(Task):

    letters = ("A", "B")

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "validation"], "BoolQ split must be train|validation"
        self.ds = load_dataset("google/boolq", split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        passage = row["passage"]
        question_text = row["question"]
        answer = row["answer"]  # True or False
        choices = ["No", "Yes"]
        answer_idx = 1 if answer else 0
        answer_string = self.letters[answer_idx]
        # create and return the Conversation object
        question = f"Passage: {passage}\n\nQuestion: {question_text}"
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
        assert assistant_response in self.letters, f"BoolQ answer {assistant_response} is expected to be one of {self.letters}"
        assistant_message = conversation['messages'][-1]['content']
        return assistant_response == assistant_message
