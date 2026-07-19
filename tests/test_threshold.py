"""Threshold-sweep tests against tiny hand-computed fixtures.

The refusal convention must exactly complement is_answerable (answer on
score >= t, refuse on score < t), so the boundary case is tested explicitly.
"""

from evals.threshold import best_threshold, refusal_rate, sweep


def test_refusal_rate_counts_scores_below_threshold():
    # 0.5 < 0.6 -> refused; 0.7 and 0.9 answer.
    assert refusal_rate([0.9, 0.7, 0.5], 0.6) == 1 / 3


def test_refusal_rate_boundary_score_answers():
    # is_answerable uses score >= threshold, so a score exactly at the
    # threshold is answered, not refused.
    assert refusal_rate([0.6], 0.6) == 0.0


def test_refusal_rate_empty_scores():
    assert refusal_rate([], 0.6) == 0.0


def test_refusal_rate_sentinel_always_refused():
    # The runner records -1.0 for empty retrieval; every threshold refuses it.
    assert refusal_rate([-1.0], 0.2) == 1.0


def test_sweep_rows_hand_computed():
    rows = sweep(
        in_domain_scores=[0.9, 0.7, 0.5],
        out_of_domain_scores=[0.4, 0.6],
        thresholds=[0.45, 0.65],
    )
    assert rows == [
        # 0.45: no in-domain below; only 0.4 of the OOD pair is below.
        {"threshold": 0.45, "false_refusal_rate": 0.0, "ood_refusal_rate": 0.5},
        # 0.65: in-domain 0.5 is below; both OOD scores are below.
        {"threshold": 0.65, "false_refusal_rate": 1 / 3, "ood_refusal_rate": 1.0},
    ]


def test_best_threshold_picks_highest_within_budget():
    rows = sweep([0.9, 0.7, 0.5], [0.4, 0.6], thresholds=[0.45, 0.55, 0.65])
    # Zero budget: 0.55 and 0.65 both refuse the 0.5 in-domain score, so the
    # only eligible row is 0.45.
    assert best_threshold(rows, max_false_refusal=0.0)["threshold"] == 0.45
    # A 1/3 budget admits every row; the highest threshold wins.
    assert best_threshold(rows, max_false_refusal=1 / 3)["threshold"] == 0.65


def test_best_threshold_none_when_budget_unmeetable():
    rows = sweep([0.3], [0.2], thresholds=[0.4, 0.5])  # every row refuses 100%
    assert best_threshold(rows, max_false_refusal=0.5) is None
