"""Dataset loader tests: BEIR corpus parsing and the loader registry.

Download/extraction is deliberately untested here (network); parsing is the
logic worth pinning down.
"""

import pytest

from rag.datasets import Document, _parse_beir_corpus, load_dataset


def test_parse_beir_corpus(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"_id": "4983", "title": "T cell responses", "text": "Alpha beta gamma."}\n'
        "\n"  # blank lines are tolerated
        '{"_id": 7, "text": "Numeric id, no title."}\n',
        encoding="utf-8",
    )
    docs = list(_parse_beir_corpus(corpus))
    assert docs == [
        Document(doc_id="4983", title="T cell responses", text="Alpha beta gamma."),
        # doc_id is coerced to str (chunk IDs are strings); missing title -> ""
        Document(doc_id="7", title="", text="Numeric id, no title."),
    ]


def test_unknown_dataset_name_is_rejected():
    with pytest.raises(ValueError, match="scifact"):
        load_dataset("nope")
