"""Threshold-sweep logic for guardrail 2 (the no-answer path), pure.

Once each query's best-chunk similarity is known, sweeping the threshold is
just counting: a query is refused at threshold t iff its top score is below t
(the exact complement of `is_answerable` in rag/retrieval.py, which answers on
score >= t). Keeping this pure — lists of floats in, rates out — is what lets
it be unit-tested against tiny hand-computed fixtures; the runner
(evals/run_threshold.py) is the only part that touches the database.
"""

from collections.abc import Sequence


def refusal_rate(top_scores: Sequence[float], threshold: float) -> float:
    """Fraction of queries refused at a threshold (top score strictly below it)."""
    if not top_scores:
        return 0.0
    return sum(1 for score in top_scores if score < threshold) / len(top_scores)


def sweep(
    in_domain_scores: Sequence[float],
    out_of_domain_scores: Sequence[float],
    thresholds: Sequence[float],
) -> list[dict]:
    """One row per candidate threshold.

    false_refusal_rate: in-domain queries refused (should stay ~0).
    ood_refusal_rate: out-of-domain queries refused (should reach 1.0).
    Both are monotonically non-decreasing in the threshold, so the sweep traces
    the whole trade-off curve.
    """
    return [
        {
            "threshold": round(t, 4),
            "false_refusal_rate": refusal_rate(in_domain_scores, t),
            "ood_refusal_rate": refusal_rate(out_of_domain_scores, t),
        }
        for t in thresholds
    ]


def best_threshold(rows: Sequence[dict], max_false_refusal: float) -> dict | None:
    """The highest-threshold row whose false-refusal rate is within budget.

    Highest, because refusal rates only grow with the threshold: the top
    eligible threshold catches the most out-of-domain queries the budget
    allows. None when even the lowest candidate exceeds the budget.
    """
    eligible = [row for row in rows if row["false_refusal_rate"] <= max_false_refusal]
    return max(eligible, key=lambda row: row["threshold"]) if eligible else None
