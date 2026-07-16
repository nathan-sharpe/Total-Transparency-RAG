"""Dataset loaders.

Each dataset gets one loader function yielding Document(doc_id, title, text);
everything downstream (chunking, ingestion) is dataset-agnostic. Adding a
corpus later (e.g. NFCorpus) means writing a new loader and registering it in
DATASETS — nothing else changes.

SciFact ships in BEIR format: a zip containing corpus.jsonl (one JSON document
per line), queries.jsonl, and qrels/ relevance labels. Only the corpus is read
here; queries and qrels stay on disk for the Phase 2 golden set.
"""

import json
import logging
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import httpx

from rag.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Per-operation (connect/read) timeout for the one-time dataset download.
# httpx is used rather than urllib because it verifies TLS against certifi's
# CA bundle — stdlib urllib fails chain verification on this Windows host.
DOWNLOAD_TIMEOUT_SECONDS = 60.0


class Document(NamedTuple):
    doc_id: str
    title: str
    text: str


def _download(url: str, dest: Path) -> None:
    """Stream url to dest, via a .part file so an interrupted download never
    looks like a finished one."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    logger.info("downloading %s -> %s", url, dest)
    with httpx.stream(
        "GET", url, follow_redirects=True, timeout=DOWNLOAD_TIMEOUT_SECONDS
    ) as response:
        response.raise_for_status()
        with open(part, "wb") as out:
            for data in response.iter_bytes():
                out.write(data)
    part.replace(dest)
    logger.info("downloaded %s (%d bytes)", dest.name, dest.stat().st_size)


def _parse_beir_corpus(corpus_path: Path) -> Iterator[Document]:
    """Yield documents from a BEIR-format corpus.jsonl."""
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            yield Document(
                doc_id=str(record["_id"]),
                title=record.get("title", ""),
                text=record["text"],
            )


def ensure_scifact(settings: Settings | None = None) -> Path:
    """Download and extract SciFact if missing; return the dataset directory.

    Idempotent: corpus.jsonl existing is the completion marker, so repeat calls
    (and every ingestion run) cost one stat.
    """
    settings = settings or get_settings()
    dataset_dir = settings.data_dir / "scifact"
    corpus_path = dataset_dir / "corpus.jsonl"
    if corpus_path.exists():
        logger.debug("scifact already present at %s", dataset_dir)
        return dataset_dir

    zip_path = settings.data_dir / "scifact.zip"
    if not zip_path.exists():
        _download(settings.scifact_url, zip_path)
    logger.info("extracting %s", zip_path)
    # The archive has a top-level scifact/ directory, so extracting into
    # data_dir produces dataset_dir.
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(settings.data_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"extracted {zip_path} but {corpus_path} is missing")
    return dataset_dir


def load_scifact(settings: Settings | None = None) -> Iterator[Document]:
    """Yield SciFact documents, downloading the corpus on first use."""
    dataset_dir = ensure_scifact(settings)
    yield from _parse_beir_corpus(dataset_dir / "corpus.jsonl")


DATASETS = {"scifact": load_scifact}


def load_dataset(name: str, settings: Settings | None = None) -> Iterator[Document]:
    """Look up a loader by its CLI name (`ingest.py --dataset <name>`)."""
    try:
        loader = DATASETS[name]
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}") from None
    return loader(settings)
