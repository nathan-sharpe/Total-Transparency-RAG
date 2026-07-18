"""Unit tests for the CI retrieval gate (evals/check_retrieval_gate.py)."""

from evals.check_retrieval_gate import check_gate


def make_results(recall5: float = 0.50, profile: str = "sentence-transformers") -> dict:
    return {
        "config": {"embedding_profile": profile},
        "metrics": {"recall@5": recall5, "recall@10": 0.60, "mrr": 0.45},
    }


def test_passes_above_floor():
    assert check_gate(make_results(recall5=0.50), floor=0.45) == []


def test_passes_exactly_at_floor():
    assert check_gate(make_results(recall5=0.45), floor=0.45) == []


def test_fails_below_floor():
    failures = check_gate(make_results(recall5=0.40), floor=0.45)
    assert len(failures) == 1
    assert "below the floor" in failures[0]


def test_fails_on_profile_mismatch_even_if_metric_clears_floor():
    failures = check_gate(
        make_results(recall5=0.99, profile="ollama"),
        floor=0.45,
        expect_profile="sentence-transformers",
    )
    assert len(failures) == 1
    assert "ollama" in failures[0]
    assert "sentence-transformers" in failures[0]


def test_matching_profile_passes():
    results = make_results(recall5=0.50, profile="sentence-transformers")
    assert check_gate(results, floor=0.45, expect_profile="sentence-transformers") == []


def test_fails_loudly_on_missing_metric():
    failures = check_gate({"config": {}, "metrics": {}}, floor=0.45)
    assert len(failures) == 1
    assert "malformed" in failures[0]
