"""Offline unit tests for windowing.py — no GPU, no data, run with `pytest tests/`."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import windowing as w  # noqa: E402


def test_parse_spec_roundtrip():
    for tok in ["roll10", "exp5", "cum", "single", "roll1", "exp2.5"]:
        assert w.parse_spec(tok).token == tok
    for bad in ["roll", "roll0", "exp-1", "nonsense", "roll10x"]:
        try:
            w.parse_spec(bad)
            assert False, f"{bad} should have raised"
        except ValueError:
            pass


def test_weights_sum_to_one_and_no_future_leakage():
    eligible = list(range(1931, 2021))
    for tok in ["roll10", "exp5", "cum", "single"]:
        ws = w.year_weights(1980, eligible, w.parse_spec(tok))
        assert abs(sum(ws.values()) - 1.0) < 1e-9
        assert all(y <= 1980 for y in ws), f"{tok} leaked a future year"


def test_rolling_window_is_recent_W_years_uniform():
    eligible = list(range(1931, 2021))
    ws = w.year_weights(1980, eligible, w.parse_spec("roll10"))
    assert sorted(ws) == list(range(1971, 1981))           # last 10 years
    assert all(abs(v - 0.1) < 1e-9 for v in ws.values())   # uniform


def test_truncated_early_window_renormalizes():
    eligible = list(range(1931, 2021))
    # 1933 with roll10 only has 1931,1932,1933 available -> uniform over 3.
    ws = w.year_weights(1933, eligible, w.parse_spec("roll10"))
    assert sorted(ws) == [1931, 1932, 1933]
    assert all(abs(v - 1 / 3) < 1e-9 for v in ws.values())


def test_roll_equals_cum_for_first_W_years():
    eligible = list(range(1931, 2021))
    for Y in range(1931, 1941):   # within the first W=10 years
        assert (w.year_weights(Y, eligible, w.parse_spec("roll10"))
                == w.year_weights(Y, eligible, w.parse_spec("cum")))


def test_exp_decays_toward_recent():
    eligible = list(range(1931, 2021))
    ws = w.year_weights(1980, eligible, w.parse_spec("exp5"))
    assert ws[1980] > ws[1975] > ws[1970]                  # recency-weighted
    # half-life 5: 1975 should be ~half of 1980's weight (pre-normalization ratio holds post-norm)
    assert abs(ws[1975] / ws[1980] - 0.5) < 1e-6


def test_allocate_per_year_uniform_gives_X_each():
    eligible = list(range(1931, 2021))
    ws = w.year_weights(1980, eligible, w.parse_spec("roll10"))
    pools = {y: 40000 for y in ws}
    counts = w.allocate_counts(ws, pools, per_year=4000)
    assert all(c == 4000 for c in counts.values())
    assert sum(counts.values()) == 40000


def test_allocate_total_n_preserves_budget():
    eligible = list(range(1931, 2021))
    ws = w.year_weights(1980, eligible, w.parse_spec("roll10"))
    pools = {y: 40000 for y in ws}
    counts = w.allocate_counts(ws, pools, total_n=12345)
    assert sum(counts.values()) == 12345                   # largest-remainder is exact


def test_allocate_clips_to_pool_sizes():
    eligible = list(range(1931, 2021))
    ws = w.year_weights(1980, eligible, w.parse_spec("roll10"))
    pools = {y: 100 for y in ws}                           # tiny pools
    counts = w.allocate_counts(ws, pools, per_year=4000)
    assert all(c <= 100 for c in counts.values())


def test_allocate_requires_exactly_one_budget():
    ws = {1980: 1.0}
    for kwargs in [{}, {"total_n": 10, "per_year": 10}]:
        try:
            w.allocate_counts(ws, {1980: 100}, **kwargs)
            assert False, "should require exactly one of total_n/per_year"
        except ValueError:
            pass


def test_stable_order_nested_across_levels():
    # The linchpin: first-k of a year's order must be a prefix-nested subset as k grows.
    order = w.stable_order(1950, 40000, seed=0)
    assert sorted(order) == list(range(40000))             # a true permutation
    for k_small, k_big in [(5000, 10000), (10000, 20000), (20000, 40000)]:
        assert set(order[:k_small]).issubset(set(order[:k_big]))


def test_stable_order_deterministic_and_year_keyed():
    assert w.stable_order(1950, 1000) == w.stable_order(1950, 1000)   # reproducible
    assert w.stable_order(1950, 1000) != w.stable_order(1951, 1000)   # year-specific


def test_experiment_name_convention():
    spec = w.parse_spec("roll10")
    assert w.experiment_name(1950, "nanochat", spec, 40000, False) == \
        "policy_1950_nanochat_roll10_n40000"
    assert w.experiment_name(1950, "talkie", spec, 5000, True) == \
        "policy_1950_talkie_roll10_n5000_sft"
