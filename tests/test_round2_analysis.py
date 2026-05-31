"""Offline unit tests for analysis/round2_report.py — pure stats, no data/GPU."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import round2_report as r2  # noqa: E402


def test_parse_name():
    c = r2.parse_name("policy_1950_nanochat_roll10_n40000")
    assert c == {"year": 1950, "backend": "nanochat", "spec": "roll10",
                 "x": 40000, "sft": False, "name": "policy_1950_nanochat_roll10_n40000"}
    c = r2.parse_name("policy_1935_talkie_roll10_n5000_sft")
    assert c["sft"] is True and c["x"] == 5000 and c["year"] == 1935
    assert r2.parse_name("not_a_window_name") is None


def test_quadratic_fit_recovers_known_parabola():
    # y = -2x^2 + 3x + 1
    xs = list(range(-5, 6))
    ys = [-2 * x ** 2 + 3 * x + 1 for x in xs]
    a, b, c = r2.quadratic_fit(xs, ys)
    assert abs(a - (-2)) < 1e-6 and abs(b - 3) < 1e-6 and abs(c - 1) < 1e-6


def test_is_n_shape_true_for_peak_at_zero():
    # Inverted-U peaking at lb=0 — the hypothesis shape.
    xs = list(range(-10, 11))
    ys = [-(x ** 2) / 100 + 0.5 for x in xs]   # concave, vertex at 0
    assert r2.is_n_shape(xs, ys) is True


def test_is_n_shape_false_for_monotone_rise():
    # A pure ramp (Round-1-style, no post-enactment peak) is NOT an n-shape.
    xs = list(range(-10, 1))
    ys = [0.02 * x + 0.3 for x in xs]
    assert r2.is_n_shape(xs, ys) is False


def test_is_n_shape_false_for_u_shape():
    # Convex (right-side-up U) must not count as an inverted-U.
    xs = list(range(-10, 11))
    ys = [(x ** 2) / 100 + 0.2 for x in xs]
    assert r2.is_n_shape(xs, ys) is False


def test_is_n_shape_false_when_peak_far_from_zero():
    # Concave but peaking at lb=+8 (long after enactment) — fails the peak_tol.
    xs = list(range(-10, 11))
    ys = [-((x - 8) ** 2) / 100 + 0.5 for x in xs]
    assert r2.is_n_shape(xs, ys, peak_tol=3.0) is False


def test_policy_trajectories_symmetric_window():
    # One policy enacted 1950, scored by models 1942..1958 -> lookback -8..+8.
    by_year = {y: {"P": (0.5, 0.0, 1950)} for y in range(1942, 1959)}
    traj = r2.policy_trajectories(by_year, lo=-10, hi=10)
    lbs = [lb for lb, _ in traj["P"]]
    assert min(lbs) == -8 and max(lbs) == 8
