"""
The ARC dataset from Allen AI.
https://huggingface.co/datasets/allenai/ai2_arc
"""

from datasets import load_dataset
from tasks.common import Task, render_mc

class ARC(Task):

    letters = ("Q", "X", "Z", "J")

    def __init__(self, subset, split, cache_dir=None, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], "ARC subset must be ARC-Easy or ARC-Challenge"
        assert split in ["train", "validation", "test"], "ARC split must be train|validation|test"
        self.ds = load_dataset("allenai/ai2_arc", subset, split=split, cache_dir=cache_dir).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"] # the question text
        choices = row["choices"]["text"] # the text of each choice
        native_answer = row["answerKey"] # e.g. "A" or "1" (varies by ARC row)
        native_letters = row["choices"]["label"] # e.g. ["A","B","C","D"] or ["1","2","3","4"]
        assert native_answer in native_letters, f"ARC answer {native_answer} must be one of {native_letters}"
        # Remap native labels (A/B/C/D or 1/2/3/4) to our scheme by POSITION.
        # Works for any native label set; we just preserve ordering.
        if len(native_letters) == 4:
            native_to_new = dict(zip(native_letters, self.letters))
            letters = list(self.letters)
            answer_string = native_to_new[native_answer]
        else:
            # Rare: some ARC questions have !=4 choices. Skip remap (keep native letters).
            letters = native_letters
            answer_string = native_answer
        # create and return the Conversation object
        user_message = render_mc(question, letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string}
        ]
        conversation = {
            "messages": messages,
            "letters": letters, # useful during evaluation, so we can narrow and clamp the assistant prediction to one of the letters
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        # the assert here is not strictly speaking needed, but currently the way we eval, we expect this to be true
        # I'm going to leave the assert here to prevent footguns, but possibly in the future can remove it.
        assert assistant_response in conversation['letters'], f"ARC answer {assistant_response} is expected to be one of {conversation['letters']}"
        assistant_message = conversation['messages'][-1]['content'] # e.g. "A"
        return assistant_response == assistant_message
