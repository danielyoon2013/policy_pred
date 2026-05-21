"""
The RACE dataset (Reading Comprehension from Examinations).
https://huggingface.co/datasets/ehovy/race

Passage-based multiple choice reading comprehension collected from
English exams for Chinese middle and high school students.
Tests reasoning from provided context rather than memorized knowledge.
"""

from datasets import load_dataset
from tasks.common import Task, render_mc

class RACE(Task):

    letters = ("Q", "X", "Z", "J")

    def __init__(self, subset, split, cache_dir=None, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["middle", "high"], "RACE subset must be middle|high"
        assert split in ["train", "validation", "test"], "RACE split must be train|validation|test"
        self.ds = load_dataset("ehovy/race", subset, split=split, cache_dir=cache_dir).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        # Prepend passage to question for context-based reasoning
        question = f"Read the following passage and answer the question.\n\nPassage: {row['article']}\n\n{row['question']}"
        choices = row["options"]  # list of 4 choices
        native_answer = row["answer"]  # A/B/C/D in the dataset
        # Remap native A/B/C/D to our letter scheme Q/X/Z/J
        remap = dict(zip("ABCD", self.letters))
        answer_string = remap[native_answer]
        user_message = render_mc(question, self.letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string}
        ]
        return {"messages": messages, "letters": self.letters}

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in self.letters
        return assistant_response == conversation['messages'][-1]['content']
