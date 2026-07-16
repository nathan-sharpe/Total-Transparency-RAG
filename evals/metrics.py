"""Retrieval metrics, hand-built and pure.

Every function takes a *ranked list of retrieved IDs* (best first) and a
collection of relevant IDs, and returns a float. The mapping from retrieved
chunks to IDs — the credit rule — happens in the runner: it passes each
retrieved chunk's doc_id, in rank order, so `k` is measured in chunks and a
chunk hits if its doc_id is relevant. Keeping the metrics generic over strings
is what lets them be unit-tested against tiny hand-computed fixtures.
"""

from collections.abc import Collection, Iterable, Sequence


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Collection[str], k: int) -> float:
    """Fraction of relevant IDs found within the top-k retrieved.

    Ceiling on whole-system quality: an answer can't cite what retrieval never
    surfaced. Duplicate retrieved IDs in the top-k collapse (a doc found twice
    is still one relevant doc found).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    found = relevant & set(retrieved_ids[:k])
    return len(found) / len(relevant)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Collection[str]) -> float:
    """1 / (rank of the first relevant ID), or 0.0 if none is retrieved.

    Rewards putting a relevant result early — LLMs attend more to earlier
    context, so position, not just presence, matters.
    """
    relevant = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def mean_recall_at_k(
    results: Iterable[tuple[Sequence[str], Collection[str]]], k: int
) -> float:
    """Mean recall@k over (retrieved_ids, relevant_ids) pairs."""
    values = [recall_at_k(retrieved, relevant, k) for retrieved, relevant in results]
    return sum(values) / len(values) if values else 0.0


def mrr(results: Iterable[tuple[Sequence[str], Collection[str]]]) -> float:
    """Mean reciprocal rank over (retrieved_ids, relevant_ids) pairs."""
    values = [reciprocal_rank(retrieved, relevant) for retrieved, relevant in results]
    return sum(values) / len(values) if values else 0.0
