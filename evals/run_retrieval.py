"""Tier-1 retrieval evaluation: run the golden set through rag/retrieval.py and
report recall@k and MRR.

    python evals/run_retrieval.py [--split test] [--limit N] [--out PATH]

Imports the pipeline directly (no HTTP), so it measures the same retrieval code
the API serves. Writes a JSON results file (metrics + a config snapshot) that
the Phase 5 CI gate will parse; also prints a human summary.

Credit rule: labels are document-level, retrieval is chunk-level, so each
retrieved chunk contributes its doc_id in rank order and a chunk hits if its
doc_id is in the query's relevant set. `k` is therefore counted in chunks.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Support both `python evals/run_retrieval.py` (roadmap demo command) and
# `python -m evals.run_retrieval`: the script form puts evals/ on sys.path
# rather than the repo root, so the package imports below would fail without
# putting the repo root back on the path first.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.golden import OUT_OF_DOMAIN_QUERIES, load_golden_set  # noqa: E402
from evals.metrics import mean_recall_at_k, mrr  # noqa: E402
from rag.config import get_settings  # noqa: E402
from rag.db import connect  # noqa: E402
from rag.embedding import get_embedder  # noqa: E402
from rag.retrieval import is_answerable, retrieve, verify_corpus_compatible  # noqa: E402

logger = logging.getLogger("evals.run_retrieval")

# The k values reported. RETRIEVE_K (>= max) is how many chunks we pull per
# query so every reported cut-off is covered by one retrieval call.
EVAL_KS = (5, 10)
RETRIEVE_K = max(EVAL_KS)

DEFAULT_OUT = Path("evals/results/retrieval.json")


def _corpus_counts(conn) -> tuple[int, int]:
    chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    docs = conn.execute("SELECT count(DISTINCT doc_id) FROM chunks").fetchone()[0]
    return chunks, docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval evaluation.")
    parser.add_argument("--split", default="test", help="qrels split (default: test)")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N queries")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="results JSON path")
    args = parser.parse_args()

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # The retrieval module logs one INFO line per query; at ~300 queries that
    # buries the summary. Lift it to WARNING for the duration of the run.
    logging.getLogger("rag.retrieval").setLevel(logging.WARNING)

    settings = get_settings()
    embedder = get_embedder(settings)

    golden = load_golden_set(split=args.split, settings=settings)
    if args.limit is not None:
        golden = golden[: args.limit]

    started = time.monotonic()
    with connect(settings) as conn:
        verify_corpus_compatible(conn, embedder)
        corpus_chunks, corpus_docs = _corpus_counts(conn)

        # In-domain: recall/MRR, plus how often retrieval clears the no-answer bar.
        graded: list[tuple[list[str], list[str]]] = []
        in_domain_answerable = 0
        for gq in golden:
            chunks = retrieve(conn, embedder, gq.query, k=RETRIEVE_K, settings=settings)
            retrieved_doc_ids = [c.doc_id for c in chunks]  # rank order, not deduped
            graded.append((retrieved_doc_ids, gq.relevant_doc_ids))
            if is_answerable(chunks, settings.no_answer_threshold):
                in_domain_answerable += 1

        # Out-of-domain: these have no relevant doc; guardrail 2 should refuse.
        ood_refused = 0
        for query in OUT_OF_DOMAIN_QUERIES:
            chunks = retrieve(conn, embedder, query, k=RETRIEVE_K, settings=settings)
            if not is_answerable(chunks, settings.no_answer_threshold):
                ood_refused += 1

    elapsed = time.monotonic() - started

    metrics = {f"recall@{k}": mean_recall_at_k(graded, k) for k in EVAL_KS}
    metrics["mrr"] = mrr(graded)
    metrics["num_queries"] = len(graded)

    n_ood = len(OUT_OF_DOMAIN_QUERIES)
    no_answer = {
        "threshold": settings.no_answer_threshold,
        "in_domain_answerable_rate": in_domain_answerable / len(graded) if graded else 0.0,
        "out_of_domain_refusal_rate": ood_refused / n_ood if n_ood else 0.0,
        "num_out_of_domain": n_ood,
    }

    results = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed, 1),
        "config": {
            "embedding_profile": settings.embedding_profile,
            "embedding_model": embedder.model_name,
            "embedding_dimension": embedder.dimension,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "retrieve_k": RETRIEVE_K,
            "corpus_chunks": corpus_chunks,
            "corpus_docs": corpus_docs,
            "split": args.split,
        },
        "metrics": metrics,
        "no_answer": no_answer,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"\nRetrieval eval — {len(graded)} queries, split={args.split}")
    print(f"  recall@5   {metrics['recall@5']:.4f}")
    print(f"  recall@10  {metrics['recall@10']:.4f}")
    print(f"  MRR        {metrics['mrr']:.4f}")
    print(f"  in-domain answerable @ thr {no_answer['threshold']:.2f}: "
          f"{no_answer['in_domain_answerable_rate']:.1%}")
    print(f"  out-of-domain refusal    @ thr {no_answer['threshold']:.2f}: "
          f"{no_answer['out_of_domain_refusal_rate']:.1%} ({ood_refused}/{n_ood})")
    print(f"  elapsed    {elapsed:.1f}s")
    print(f"  wrote      {args.out}")


if __name__ == "__main__":
    main()
