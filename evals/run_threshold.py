"""Guardrail-2 threshold sweep: find where the no-answer bar should sit.

    python evals/run_threshold.py [--split test] [--limit N] [--out PATH]

Collects each golden query's best-chunk similarity (top-1 retrieval) and the
same for the hand-authored out-of-domain queries, then sweeps a threshold grid
and reports, per threshold, the false-refusal rate (in-domain refused) against
the out-of-domain refusal rate. The recommended operating points — the highest
threshold within each false-refusal budget — are printed and written to the
results JSON alongside the raw score distributions, so the choice is auditable.

Scores are similarity-scale-specific: a sweep is only valid for the embedding
profile (and corpus) it was run against. The config snapshot in the output
records both.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.golden import OUT_OF_DOMAIN_QUERIES, load_golden_set  # noqa: E402
from evals.threshold import best_threshold, sweep  # noqa: E402
from rag.config import get_settings  # noqa: E402
from rag.db import connect  # noqa: E402
from rag.embedding import get_embedder  # noqa: E402
from rag.retrieval import retrieve, verify_corpus_compatible  # noqa: E402

logger = logging.getLogger("evals.run_threshold")

# 0.20-0.80 in steps of 0.01 comfortably brackets both observed profiles'
# similarity scales (see EVALS.md).
THRESHOLD_GRID = [round(0.20 + 0.01 * i, 2) for i in range(61)]

# False-refusal budgets the recommendation is computed for: how many in-domain
# queries we are willing to wrongly refuse in exchange for a stricter gate.
FALSE_REFUSAL_BUDGETS = (0.0, 0.01, 0.02, 0.05)

DEFAULT_OUT = Path("evals/results/threshold.json")


def collect_top_scores(conn, embedder, queries: list[str], settings) -> list[float]:
    """Best-chunk similarity per query; -1.0 if retrieval returned nothing
    (refused at every threshold, matching is_answerable on an empty list)."""
    scores = []
    for query in queries:
        chunks = retrieve(conn, embedder, query, k=1, settings=settings)
        scores.append(chunks[0].score if chunks else -1.0)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep the no-answer threshold.")
    parser.add_argument("--split", default="test", help="qrels split (default: test)")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N queries")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="results JSON path")
    args = parser.parse_args()

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    logging.getLogger("rag.retrieval").setLevel(logging.WARNING)

    settings = get_settings()
    embedder = get_embedder(settings)

    golden = load_golden_set(split=args.split, settings=settings)
    if args.limit is not None:
        golden = golden[: args.limit]

    started = time.monotonic()
    with connect(settings) as conn:
        verify_corpus_compatible(conn, embedder)
        in_domain = collect_top_scores(
            conn, embedder, [gq.query for gq in golden], settings
        )
        out_of_domain = collect_top_scores(
            conn, embedder, list(OUT_OF_DOMAIN_QUERIES), settings
        )
    elapsed = time.monotonic() - started

    curve = sweep(in_domain, out_of_domain, THRESHOLD_GRID)
    recommendations = {
        str(budget): best_threshold(curve, budget) for budget in FALSE_REFUSAL_BUDGETS
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
            "current_threshold": settings.no_answer_threshold,
            "split": args.split,
        },
        "scores": {
            "in_domain": in_domain,
            "out_of_domain": out_of_domain,
        },
        "curve": curve,
        "recommendations": recommendations,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    in_sorted = sorted(in_domain)
    print(f"\nThreshold sweep — {len(in_domain)} in-domain / {len(out_of_domain)} out-of-domain")
    print(f"  in-domain top scores   min {in_sorted[0]:.4f}"
          f"  p5 {in_sorted[len(in_sorted) // 20]:.4f}"
          f"  median {in_sorted[len(in_sorted) // 2]:.4f}  max {in_sorted[-1]:.4f}")
    print(f"  out-of-domain scores   max {max(out_of_domain):.4f}")
    for budget, row in recommendations.items():
        label = f"budget {float(budget):.0%} false refusals"
        if row is None:
            print(f"  {label}: no threshold on the grid qualifies")
        else:
            print(f"  {label} -> threshold {row['threshold']:.2f} "
                  f"(refuses {row['ood_refusal_rate']:.0%} of out-of-domain)")
    print(f"  elapsed    {elapsed:.1f}s")
    print(f"  wrote      {args.out}")


if __name__ == "__main__":
    main()
