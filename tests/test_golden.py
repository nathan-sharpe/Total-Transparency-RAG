"""Golden-set loader tests over tiny on-disk fixtures.

A temp SciFact-shaped directory is built so load_golden_set does its real
qrels/queries join without downloading anything (ensure_scifact treats an
existing corpus.jsonl as the completion marker).
"""

from pathlib import Path

import pytest

from evals.golden import _load_qrels, _load_query_texts, load_golden_set
from rag.config import Settings


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_scifact(tmp_path: Path, queries: str, qrels: str) -> Settings:
    root = tmp_path / "scifact"
    write(root / "corpus.jsonl", '{"_id": "1", "title": "t", "text": "body"}\n')
    write(root / "queries.jsonl", queries)
    write(root / "qrels" / "test.tsv", qrels)
    return Settings(
        _env_file=None, postgres_user="t", postgres_password="t", data_dir=tmp_path
    )


def test_load_qrels_groups_and_filters_by_score(tmp_path: Path):
    path = tmp_path / "test.tsv"
    write(path, "query-id\tcorpus-id\tscore\n1\t100\t1\n1\t200\t1\n3\t300\t1\n5\t400\t0\n")
    qrels = _load_qrels(path)
    assert qrels == {"1": ["100", "200"], "3": ["300"]}  # score 0 row dropped


def test_load_qrels_rejects_bad_header(tmp_path: Path):
    path = tmp_path / "bad.tsv"
    write(path, "wrong\theader\there\n1\t2\t1\n")
    with pytest.raises(ValueError):
        _load_qrels(path)


def test_load_query_texts_maps_id_to_text(tmp_path: Path):
    path = tmp_path / "queries.jsonl"
    write(path, '{"_id": "1", "text": "first"}\n{"_id": "2", "text": "second"}\n')
    assert _load_query_texts(path) == {"1": "first", "2": "second"}


def test_load_golden_set_joins_and_sorts(tmp_path: Path):
    queries = (
        '{"_id": "10", "text": "ten"}\n'
        '{"_id": "2", "text": "two"}\n'
        '{"_id": "99", "text": "unlabeled, not in qrels"}\n'
    )
    qrels = "query-id\tcorpus-id\tscore\n10\t500\t1\n2\t600\t1\n2\t601\t1\n"
    settings = make_scifact(tmp_path, queries, qrels)

    golden = load_golden_set(split="test", settings=settings)

    # Numeric-ascending order; query 99 excluded (no qrels label).
    assert [g.query_id for g in golden] == ["2", "10"]
    assert golden[0].relevant_doc_ids == ["600", "601"]  # labels are a list
    assert golden[1] == golden[1].__class__("10", "ten", ["500"])


def test_load_golden_set_skips_qrels_ids_missing_text(tmp_path: Path):
    # qrels references query 7, but queries.jsonl has no such id -> skipped.
    queries = '{"_id": "2", "text": "two"}\n'
    qrels = "query-id\tcorpus-id\tscore\n2\t600\t1\n7\t700\t1\n"
    settings = make_scifact(tmp_path, queries, qrels)

    golden = load_golden_set(split="test", settings=settings)
    assert [g.query_id for g in golden] == ["2"]
