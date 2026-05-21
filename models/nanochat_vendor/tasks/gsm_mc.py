"""
GSM-MC: Multiple-choice conversion of GSM8K.
Zhang et al., 2024 — github.com/Geralt-Targaryen/MC-Evaluation

4-choice math word problems with distractors collected from real model mistakes
across 60 open-source models. Strongly correlated with generative GSM8K scores.

Data format (from gsm8k-mc/test.jsonl inside data.tar.gz):
    {"Question": "...", "A": "...", "B": "...", "C": "...", "D": "...", "Answer": "B"}
"""

import json
from tasks.common import Task, render_mc


class GSMMC(Task):

    letters = ("Q", "X", "Z", "J")

    def __init__(self, data_file, **kwargs):
        super().__init__(**kwargs)
        with open(data_file, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.data)

    def get_example(self, index):
        row = self.data[index]
        question = row["Question"]
        choices = [row["A"], row["B"], row["C"], row["D"]]
        native_answer = row["Answer"]  # A/B/C/D in raw data
        # Remap native A/B/C/D to our letter scheme
        remap = dict(zip("ABCD", self.letters))
        assert native_answer in remap, f"GSM-MC native answer {native_answer} must be A/B/C/D"
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
