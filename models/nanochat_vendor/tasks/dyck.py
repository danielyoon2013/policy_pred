"""
Dyck Language evaluation task (BigBench, BBH-style generative exact match).

Tests whether a model can track hierarchical bracket-matching structure.
Given an incomplete Dyck-4 sequence (using bracket types: () [] {} <>),
the model must produce the correct closing brackets.

Example:
    Input:  { [ [ [ { [ ] } ] ]
    Output: ] }

This is the standard evaluation format used by BBH (Suzgun et al., 2023),
HELM, and lm-evaluation-harness. Generative with exact match scoring.

To preview examples:
    python -m tasks.dyck
"""

from datasets import load_dataset
from tasks.common import Task


def normalize_brackets(s):
    """Extract only bracket characters, space-separated.

    Strips any non-bracket text the model might generate around the answer.
    E.g. "The answer is ] }" -> "] }"
         "] }\\n\\n" -> "] }"
    """
    return ' '.join(c for c in s if c in '()[]<>{}')


class DyckLanguage(Task):

    def __init__(self, split="default", **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("hails/bigbench", "dyck_languages_zero_shot", split=split)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        target = row["targets"][0]  # e.g., "] }"
        return {
            "messages": [
                {"role": "user", "content": row["inputs"]},
                {"role": "assistant", "content": target},
            ]
        }

    def evaluate(self, conversation, assistant_response):
        """Exact match on normalized bracket sequences."""
        ref = conversation["messages"][-1]["content"]
        pred_brackets = normalize_brackets(assistant_response)
        ref_brackets = normalize_brackets(ref)
        return int(pred_brackets == ref_brackets)

    def reward(self, conversation, assistant_response):
        return float(self.evaluate(conversation, assistant_response))


if __name__ == "__main__":
    task = DyckLanguage()
    print(f"Dyck Language: {len(task)} examples")
    for i in range(5):
        ex = task.get_example(i)
        user_msg = ex["messages"][0]["content"]
        # Extract just the bracket sequence from the prompt
        bracket_input = user_msg.split("Input: ")[-1].split("\nOutput:")[0]
        target = ex["messages"][1]["content"]
        print(f"\n  Input:  {bracket_input}")
        print(f"  Target: {target}")
