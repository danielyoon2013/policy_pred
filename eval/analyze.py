"""Build P(implemented | year) trajectories from elicit outputs.

For each policy, aggregate per-year probe scores into a trajectory; write a CSV and a
plot. Useful headline metrics: argmax-year (does the trajectory peak at or near
implementation_year?) and rank-correlation between year and probability.
"""


def build_trajectories() -> None:
    raise NotImplementedError
