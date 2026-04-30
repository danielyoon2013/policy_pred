"""Run policy probes against a year-Y model checkpoint, save log-probs.

Loads the year-Y SFT checkpoint via the same backend type as BASE_MODEL_SPEC,
iterates the catalog, builds probes via probes.build_probes, scores them with
backend.score_continuations, and writes per-probe scores to policy_eval_path.
"""


def run_all_policies(year: int) -> None:
    raise NotImplementedError
