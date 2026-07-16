"""Retrieval metric tests against tiny hand-computed fixtures.

Every expected value here is worked out by hand from the ranked list and the
relevant set, so a regression in the metric math is caught by eye.
"""

import pytest

from evals.metrics import mean_recall_at_k, mrr, recall_at_k, reciprocal_rank


def test_recall_at_k_partial_and_full():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d"}
    assert recall_at_k(retrieved, relevant, k=1) == 0.0  # top {a}
    assert recall_at_k(retrieved, relevant, k=2) == 0.5  # top {a,b} -> found b
    assert recall_at_k(retrieved, relevant, k=4) == 1.0  # both found


def test_recall_at_k_singleton_relevant():
    assert recall_at_k(["a", "b"], ["b"], k=2) == 1.0
    assert recall_at_k(["a", "b"], ["b"], k=1) == 0.0


def test_recall_at_k_collapses_duplicate_ids():
    # A doc retrieved twice within the cut-off is still one relevant doc found.
    assert recall_at_k(["a", "a", "b"], {"a"}, k=2) == 1.0


def test_recall_at_k_empty_relevant_is_zero():
    assert recall_at_k(["a", "b"], [], k=2) == 0.0


def test_recall_at_k_rejects_nonpositive_k():
    with pytest.raises(ValueError):
        recall_at_k(["a"], {"a"}, k=0)


def test_reciprocal_rank_position():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)
    # First relevant wins when several are relevant.
    assert reciprocal_rank(["a", "b", "c"], {"b", "c"}) == 0.5


def test_reciprocal_rank_none_retrieved():
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_mrr_averages_reciprocal_ranks():
    results = [
        (["a", "b", "c"], {"a"}),  # rr = 1.0
        (["a", "b", "c"], {"c"}),  # rr = 1/3
    ]
    assert mrr(results) == pytest.approx((1.0 + 1 / 3) / 2)


def test_mean_recall_at_k_averages_queries():
    results = [
        (["a", "b"], {"a"}),  # recall@2 = 1.0
        (["a", "b"], {"z"}),  # recall@2 = 0.0
    ]
    assert mean_recall_at_k(results, k=2) == 0.5


def test_aggregate_helpers_handle_empty_input():
    assert mrr([]) == 0.0
    assert mean_recall_at_k([], k=5) == 0.0
