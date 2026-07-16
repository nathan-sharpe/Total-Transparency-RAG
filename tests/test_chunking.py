"""Chunker unit tests: deterministic IDs, overlap behavior, edge cases.

The windowing tests use numbered words (w0 w1 w2 ...) so expected chunks can
be written out by hand and checked by eye.
"""

import pytest

from rag.chunking import Chunk, chunk_document, chunk_text, make_chunk_id


def numbered_words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_make_chunk_id_format():
    assert make_chunk_id("doc42", 0) == "doc42::0"
    assert make_chunk_id("doc42", 17) == "doc42::17"


def test_short_text_is_a_single_chunk():
    assert chunk_text("one two three", chunk_size=10, chunk_overlap=2) == ["one two three"]


def test_text_exactly_chunk_size_is_a_single_chunk():
    text = numbered_words(5)
    assert chunk_text(text, chunk_size=5, chunk_overlap=2) == [text]


def test_sliding_window_and_overlap():
    # 12 words, size 5, overlap 2 -> stride 3 -> windows start at 0, 3, 6, 9.
    pieces = chunk_text(numbered_words(12), chunk_size=5, chunk_overlap=2)
    assert pieces == [
        "w0 w1 w2 w3 w4",
        "w3 w4 w5 w6 w7",
        "w6 w7 w8 w9 w10",
        "w9 w10 w11",
    ]
    # Consecutive chunks share exactly the overlap.
    for left, right in zip(pieces, pieces[1:], strict=False):
        assert left.split()[-2:] == right.split()[:2]


def test_zero_overlap_partitions_exactly():
    pieces = chunk_text(numbered_words(10), chunk_size=5, chunk_overlap=0)
    assert pieces == ["w0 w1 w2 w3 w4", "w5 w6 w7 w8 w9"]


def test_no_words_are_lost():
    original = numbered_words(23).split()
    pieces = chunk_text(numbered_words(23), chunk_size=7, chunk_overlap=3)
    covered = {word for piece in pieces for word in piece.split()}
    assert covered == set(original)
    assert pieces[-1].split()[-1] == "w22"  # final chunk reaches the end


def test_whitespace_is_normalized():
    assert chunk_text("  spaced \t out\n\nwords ", chunk_size=10, chunk_overlap=0) == [
        "spaced out words"
    ]


def test_empty_and_blank_text_yield_no_chunks():
    assert chunk_text("", chunk_size=5, chunk_overlap=1) == []
    assert chunk_text("   \n\t ", chunk_size=5, chunk_overlap=1) == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (-1, 0), (5, 5), (5, 6), (5, -1)],
)
def test_invalid_parameters_are_rejected(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        chunk_text("some words here", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_chunk_document_ids_are_deterministic():
    chunks = chunk_document("doc42", numbered_words(12), chunk_size=5, chunk_overlap=2)
    assert [c.chunk_id for c in chunks] == ["doc42::0", "doc42::1", "doc42::2", "doc42::3"]
    assert [c.chunk_index for c in chunks] == [0, 1, 2, 3]
    assert all(c.doc_id == "doc42" for c in chunks)
    # Same input, same output — run it twice.
    assert chunks == chunk_document("doc42", numbered_words(12), chunk_size=5, chunk_overlap=2)


def test_chunk_document_single_chunk_doc():
    chunks = chunk_document("d1", "a short abstract", chunk_size=200, chunk_overlap=40)
    assert chunks == [Chunk(chunk_id="d1::0", doc_id="d1", chunk_index=0, text="a short abstract")]
