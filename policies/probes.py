"""Build MC probes for a policy.

For each policy, generate N paraphrased multiple-choice probes of the form:
    'Among the following, which policy is most likely to be enacted in the coming
    years?' [policy.description, distractor1, distractor2, distractor3]
Distractors must be plausible-but-fictional within the era so the test isn't trivial.
Average normalized log-prob on the correct option across paraphrases.
"""


def build_probes(policy: dict, n_paraphrases: int = 16) -> list[dict]:
    raise NotImplementedError
