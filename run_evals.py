"""Tier-2 generation evaluation: LLM-as-judge over the golden queries.

    python run_evals.py [--split test] [--limit N] [--out PATH]

Manual, laptop-only (needs Ollama + the ingested corpus) — never run in CI.
Results are committed to EVALS.md by hand.

Runs in TWO PASSES because the 6GB-VRAM budget fits one 7-8B model at a time:
pass 1 generates every answer with the generator model, pass 2 judges them
all with the judge model. Interleaving would force Ollama to swap models in
and out of VRAM on every query. Answers are written to disk between passes so
a judging failure never loses the (equally expensive) generation work.

Per case: retrieve -> guardrail 2 (refuse below threshold, generator never
called) -> generate -> guardrail 3b (ground citations) -> judge. Refusals
(either the guardrail's or the generator's own) are counted, not judged —
"was this refusal correct?" is what the refusal-rate numbers answer.

Judge verdicts are schema-validated with one retry (guardrail 3a, in
evals/judge.py). A case that still fails is excluded from the aggregates and
counted in `judge_failures` — garbage never enters metrics, and a non-zero
failure count is reported loudly.

Monitoring a long run: every run writes a timestamped live log (default
evals/results/run_evals.log) with elapsed + ETA on each progress line. Tail it
    Get-Content evals/results/run_evals.log -Wait -Tail 20   (PowerShell)
    tail -f evals/results/run_evals.log                      (bash)
This is more reliable than an IDE task panel, whose runtime counter can freeze
while output keeps streaming.
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
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from evals.golden import load_golden_set  # noqa: E402
from evals.judge import JudgeError, judge_answer  # noqa: E402
from rag.config import get_settings  # noqa: E402
from rag.db import connect  # noqa: E402
from rag.embedding import get_embedder  # noqa: E402
from rag.generation import (  # noqa: E402
    NO_ANSWER_RESPONSE,
    generate_answer,
    ground_citations,
)
from rag.retrieval import is_answerable, retrieve, verify_corpus_compatible  # noqa: E402

logger = logging.getLogger("evals.run_evals")

DEFAULT_OUT = Path("evals/results/generation.json")
ANSWERS_OUT = Path("evals/results/answers.json")
DEFAULT_LOG = Path("evals/results/run_evals.log")

# How often the passes emit a progress line. Each item is a multi-second LLM
# call, so every 10 is roughly a line every couple of minutes — frequent enough
# to tail live, sparse enough not to bury the run.
PROGRESS_EVERY = 10


def _log_progress(label: str, done: int, total: int, started: float) -> None:
    """One self-contained progress line: count, %, elapsed, rate, and ETA — so a
    single tailed line tells you how far along the run is and how long is left."""
    elapsed = time.monotonic() - started
    rate = done / elapsed if elapsed else 0.0
    eta = (total - done) / rate if rate else 0.0
    logger.info(
        "%s %d/%d (%.0f%%) — elapsed %s, %.1fs/item, ETA %s",
        label, done, total, 100 * done / total,
        _fmt_duration(elapsed), 1 / rate if rate else 0.0, _fmt_duration(eta),
    )


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def generate_all(golden, settings, embedder) -> list[dict]:
    """Pass 1: retrieve + generate (or refuse) for every golden query."""
    cases = []
    started = time.monotonic()
    with connect(settings) as conn:
        verify_corpus_compatible(conn, embedder)
        for i, gq in enumerate(golden, 1):
            chunks = retrieve(conn, embedder, gq.query, settings=settings)
            case = {
                "query_id": gq.query_id,
                "query": gq.query,
                "top_score": chunks[0].score if chunks else 0.0,
                "chunks": [
                    {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text,
                     "score": c.score}
                    for c in chunks
                ],
            }
            if not is_answerable(chunks, settings.no_answer_threshold):
                case.update(refused="guardrail", answer=NO_ANSWER_RESPONSE)
            else:
                result = generate_answer(gq.query, chunks, settings=settings)
                if result.answer == NO_ANSWER_RESPONSE:
                    case.update(refused="generator", answer=NO_ANSWER_RESPONSE)
                else:
                    grounded = ground_citations(
                        result.answer, [c.chunk_id for c in chunks]
                    )
                    case.update(
                        refused=None,
                        answer=grounded.answer,
                        grounded_citations=grounded.grounded_citations,
                        ungrounded_citations=grounded.ungrounded_citations,
                    )
            cases.append(case)
            if i % PROGRESS_EVERY == 0 or i == len(golden):
                _log_progress("generated", i, len(golden), started)
    return cases


def judge_all(cases: list[dict], settings) -> int:
    """Pass 2: judge every non-refused answer in place. Returns failure count."""
    from rag.retrieval import RetrievedChunk

    to_judge = [c for c in cases if c["refused"] is None]
    failures = 0
    started = time.monotonic()
    for i, case in enumerate(to_judge, 1):
        chunks = [RetrievedChunk(**ch) for ch in case["chunks"]]
        try:
            verdict = judge_answer(case["query"], chunks, case["answer"], settings)
        except JudgeError:
            logger.exception("judge failed for query %s — excluded", case["query_id"])
            failures += 1
            case["judge"] = None
            continue
        case["judge"] = verdict.model_dump()
        if i % PROGRESS_EVERY == 0 or i == len(to_judge):
            _log_progress("judged", i, len(to_judge), started)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Tier-2 generation eval.")
    parser.add_argument("--split", default="test", help="qrels split (default: test)")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N queries")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="results JSON path")
    parser.add_argument(
        "--resume-judge",
        action="store_true",
        help=(
            "skip the generation pass and judge the answers already saved in "
            f"{ANSWERS_OUT} (recovers an interrupted run without regenerating)"
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG,
        help=(
            f"live progress log, overwritten each run (default: {DEFAULT_LOG}). "
            "Tail it during a run: Get-Content <path> -Wait -Tail 20"
        ),
    )
    args = parser.parse_args()

    # Logs go to BOTH stdout and a file. The file is the reliable live view:
    # tail it with `Get-Content <log-file> -Wait` (PowerShell) / `tail -f`. Every
    # line is timestamped and progress lines carry elapsed + ETA, so unlike an
    # IDE task panel's frozen runtime counter, the file always shows real state.
    # Opened in 'w' mode so each run starts a clean log (no stale prior-run lines).
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(args.log_file, mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
    )
    # Per-query INFO lines from these modules would bury the progress lines.
    for noisy in ("rag.retrieval", "rag.generation", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    settings = get_settings()

    if args.resume_judge:
        # Recover an interrupted run: the (expensive) generation pass already
        # wrote every answer to disk, so judge those instead of regenerating.
        cases = json.loads(ANSWERS_OUT.read_text(encoding="utf-8"))
        generation_seconds = 0.0
        logger.info("resuming: judging %d saved answers from %s", len(cases), ANSWERS_OUT)
    else:
        embedder = get_embedder(settings)
        golden = load_golden_set(split=args.split, settings=settings)
        if args.limit is not None:
            golden = golden[: args.limit]

        started = time.monotonic()
        cases = generate_all(golden, settings, embedder)
        generation_seconds = time.monotonic() - started

        # Safety artifact: judging can fail without costing the generation pass.
        ANSWERS_OUT.parent.mkdir(parents=True, exist_ok=True)
        ANSWERS_OUT.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
        logger.info(
            "generation pass done in %.0fs, answers saved to %s", generation_seconds, ANSWERS_OUT
        )

    judge_started = time.monotonic()
    judge_failures = judge_all(cases, settings)
    judge_seconds = time.monotonic() - judge_started

    judged = [c for c in cases if c["refused"] is None and c.get("judge")]
    answered = [c for c in cases if c["refused"] is None]
    refusals = {
        "guardrail": sum(1 for c in cases if c["refused"] == "guardrail"),
        "generator": sum(1 for c in cases if c["refused"] == "generator"),
    }
    with_ungrounded = [c for c in answered if c["ungrounded_citations"]]

    metrics = {
        "num_queries": len(cases),
        "num_answered": len(answered),
        "num_judged": len(judged),
        "judge_failures": judge_failures,
        "refusals": refusals,
        "mean_faithfulness": (
            sum(c["judge"]["faithfulness"] for c in judged) / len(judged) if judged else None
        ),
        "mean_relevance": (
            sum(c["judge"]["relevance"] for c in judged) / len(judged) if judged else None
        ),
        "faithfulness_at_least_4_rate": (
            sum(c["judge"]["faithfulness"] >= 4 for c in judged) / len(judged)
            if judged else None
        ),
        "answers_with_ungrounded_citations": len(with_ungrounded),
        "ungrounded_citations_total": sum(
            len(c["ungrounded_citations"]) for c in answered
        ),
    }

    results = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "generation_seconds": round(generation_seconds, 1),
        "judge_seconds": round(judge_seconds, 1),
        "config": {
            "generator_model": settings.generator_model,
            "judge_model": settings.judge_model,
            "embedding_model": settings.resolved_embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "max_context_chunks": settings.max_context_chunks,
            "no_answer_threshold": settings.no_answer_threshold,
            "split": args.split,
            "limit": args.limit,
        },
        "metrics": metrics,
        # Per-case verdicts (without chunk texts) for error analysis.
        "cases": [
            {k: v for k, v in case.items() if k != "chunks"} for case in cases
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"\nGeneration eval — {len(cases)} queries, split={args.split}")
    print(f"  answered / refused (guardrail) / refused (generator): "
          f"{len(answered)} / {refusals['guardrail']} / {refusals['generator']}")
    if judged:
        print(f"  mean faithfulness   {metrics['mean_faithfulness']:.2f} / 5")
        print(f"  mean relevance      {metrics['mean_relevance']:.2f} / 5")
        print(f"  faithfulness >= 4   {metrics['faithfulness_at_least_4_rate']:.1%}")
    print(f"  ungrounded citations: {metrics['ungrounded_citations_total']} "
          f"(in {metrics['answers_with_ungrounded_citations']} answers)")
    if judge_failures:
        print(f"  !! JUDGE FAILURES: {judge_failures} case(s) excluded from metrics")
    print(f"  generation {generation_seconds:.0f}s, judging {judge_seconds:.0f}s")
    print(f"  wrote      {args.out}")


if __name__ == "__main__":
    main()
