"""SciFact golden set: labeled test queries with their relevant document IDs.

BEIR ships SciFact as queries.jsonl (all splits) plus qrels/<split>.tsv giving
the relevance labels. A query belongs to a split iff it appears in that split's
qrels file, so the test golden set is the join of qrels/test.tsv (which query,
which relevant docs) with queries.jsonl (the query text).

The credit rule (documented in EVALS.md) lives in the *runner*, not here:
labels are document-level, retrieval is chunk-level, so a retrieved chunk is
counted relevant iff its doc_id is in a query's relevant_doc_ids. This module
only supplies the labels; it keeps them as a list per query (singleton or not)
so multi-relevant metrics need no schema change.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from rag.config import Settings
from rag.datasets import ensure_scifact

logger = logging.getLogger(__name__)


class GoldenQuery(NamedTuple):
    query_id: str
    query: str
    relevant_doc_ids: list[str]


# Hand-authored queries that are clearly outside SciFact's biomedical-science
# scope. They have no relevant document, so they exercise guardrail 2 (the
# no-answer path): a healthy system should refuse them before generating.
OUT_OF_DOMAIN_QUERIES: list[str] = [
    "What is the best way to grill a ribeye steak?",
    "Who won the 2018 FIFA World Cup?",
    "How do I change a flat tire on a car?",
    "What is the capital of Australia?",
    "Write a haiku about autumn leaves.",
    "What time does the next train to Boston leave?",
    "How do I reset my email password?",
    "What are the rules of chess?",
]


def _load_qrels(qrels_path: Path) -> dict[str, list[str]]:
    """Parse a BEIR qrels TSV (header: query-id, corpus-id, score) into
    query_id -> [relevant doc_ids]. Only positively-labeled rows count."""
    relevant: dict[str, list[str]] = defaultdict(list)
    with open(qrels_path, encoding="utf-8") as f:
        header = next(f, "")
        if not header.lower().startswith("query-id"):
            raise ValueError(f"unexpected qrels header in {qrels_path}: {header!r}")
        for line in f:
            if not line.strip():
                continue
            query_id, corpus_id, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                relevant[str(query_id)].append(str(corpus_id))
    return dict(relevant)


def _load_query_texts(queries_path: Path) -> dict[str, str]:
    """Parse queries.jsonl into query_id -> query text."""
    texts: dict[str, str] = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            texts[str(record["_id"])] = record["text"]
    return texts


def load_golden_set(
    split: str = "test", settings: Settings | None = None
) -> list[GoldenQuery]:
    """Load the labeled golden set for a split, downloading SciFact if missing.

    Returned in a stable order (by numeric query_id when possible) so eval runs
    are reproducible.
    """
    dataset_dir = ensure_scifact(settings)
    qrels = _load_qrels(dataset_dir / "qrels" / f"{split}.tsv")
    texts = _load_query_texts(dataset_dir / "queries.jsonl")

    golden: list[GoldenQuery] = []
    missing = 0
    for query_id, doc_ids in qrels.items():
        text = texts.get(query_id)
        if text is None:
            missing += 1
            continue
        golden.append(GoldenQuery(query_id=query_id, query=text, relevant_doc_ids=doc_ids))
    if missing:
        logger.warning("%d qrels query ids had no text in queries.jsonl (skipped)", missing)

    golden.sort(key=lambda g: (0, int(g.query_id)) if g.query_id.isdigit() else (1, g.query_id))
    logger.info("loaded %d golden queries (split=%s)", len(golden), split)
    return golden
