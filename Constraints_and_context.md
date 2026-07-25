# RAG Project Planning Summary

Reference document compiled from planning discussions, July 2026. Intended use: context for Claude Code plan mode and as a checklist while building.

## Purpose and positioning

The project exists to fill identified portfolio gaps: public code footprint and Python-level engineering, formal evaluation methodology, and production operations depth. It is a public, Python-first RAG system that demonstrates the patterns behind prior client work in verifiable, open form.

Strict rule: zero n8n in the implementation. Everything runs through code, turning the n8n background into demonstrated architectural judgment rather than something being compensated for. That judgment is meant to show through the artifacts themselves — hand-built chunking/retrieval/metrics, git-diffable tuning experiments with before/after numbers, CI replaying the whole pipeline — rather than through an explicit comparison section. *(Revised 2026-07-25: the README originally carried a "code-first vs. n8n" comparison section per this rule; it was removed as out of place. See the Phase 6 notes in ROADMAP.md.)*

## Stack and constraints

- Python end to end: ingestion, chunking, embedding, retrieval, generation, API, evals.
- FastAPI for the API layer, uvicorn as the server (both pip-installed, pinned in requirements.txt). Endpoints are created by the developer (POST /query, GET /health; ingestion via CLI script rather than endpoint). FastAPI auto-generates an interactive Swagger UI at /docs from type-hinted endpoints; that page doubles as the demo interface.
- Local LLMs via Ollama for cost and as a talking point (local AI + Docker experience). Hardware (RTX 4050 6GB VRAM, 64GB RAM) supports 7-8B quantized models (Llama 3.1 8B, Qwen 2.5 7B) and embedding models like nomic-embed-text comfortably. On Windows, run Ollama natively on the host; containers reach it at http://host.docker.internal:11434.
- Postgres + pgvector in a container as the vector store (same architecture as Supabase, self-contained). Needs a volume mount for data persistence.
- Consider implementing core steps (chunking, retrieval scoring) by hand rather than importing everything from LangChain; "built it with and without the framework" is a strong position.

## Dataset

- Option chosen: an existing benchmark with golden labels. Recommended: SciFact (~5,200 scientific abstracts, ~300 labeled test queries, BEIR-standard, laptop-friendly). NFCorpus (~3.6K docs) as a possible second set later.
- Multi-dataset verification (BEIR suite) deferred as a future extension; current scope is optimizing one system for one corpus, measured rigorously.
- Custom golden-set generation (RAGAS synthesis + manual verification) noted as the future approach for domain-specific corpora.

## Evaluation

Two layers: retrieval (did we fetch the right chunks?) and generation (given chunks, was the answer good?).

- Golden dataset: fixed query -> relevant-chunk-IDs pairs (store labels as a list per query from day one, even when singleton, so multi-chunk cases need no schema change). SciFact ships with these labels.
- Retrieval metrics, implemented by hand (~50 lines, demonstrates understanding):
  - recall@k: fraction of relevant chunks appearing in top k. Ceiling on whole-system quality.
  - MRR: mean of 1/rank of first relevant result; captures position, which matters because LLMs attend more to earlier context.
  - nDCG noted as stretch goal for graded multi-chunk ranking.
- Generation metrics via LLM-as-judge: a code loop fills a prompt template (rubric in system prompt; query/chunks/answer interpolated; output constrained to JSON schema), calls the judge model per test case, parses verdicts, aggregates into pass rates or averaged scores. RAGAS provides ready-made metrics (faithfulness = fraction of answer claims supported by retrieved context, context precision, answer relevance).
- Local-judge caveat: an 8B judge is noisy. Document the limitation honestly; optionally re-score final results once with a cheap API model for comparison.
- Value of metrics is relative comparison under controlled conditions (same corpus, same golden set, before/after a change), not absolute cross-domain scores.
- EVALS.md: living document. Corpus and golden set description, baseline numbers, then dated changelog entries: change -> hypothesis -> numbers before/after -> conclusion. This document is the portfolio centerpiece.

## CI (GitHub Actions)

- Mechanism: .github/workflows/ci.yml; on every push GitHub runs a fresh Ubuntu VM: checkout, setup Python, install deps, ruff lint, pytest. Any failure = red X on the commit. CI checks the state of the whole repo after each push, not just changed files.
- Two-tier eval split (chosen approach):
  - Tier 1, in CI on every push: unit tests + retrieval metrics using a small CPU-friendly embedding model (sentence-transformers scale) downloaded fresh each run. Includes an eval gate: script exits nonzero if recall@5 drops below a floor.
  - Tier 2, manual on the laptop: generation/LLM-as-judge evals against local Ollama, run by choice (python run_evals.py); results committed to EVALS.md. CI never touches the laptop; git carries the write-up.
- GitHub runners cannot reach the laptop (network reachability, not authentication). Alternatives noted: hosted-API key in GitHub Secrets, or self-hosted runner (documented stretch goal).
- Doc-only pushes still trigger CI harmlessly; optional paths-ignore for **.md or [skip ci] in commit messages, not bothered with by default.

## Containers

- Dockerfile for the API app + docker-compose.yml bringing up API + pgvector Postgres with one command. Target README line: "clone, docker compose up, running locally."
- Base image = entire starting filesystem (minimal Linux + Python), e.g. FROM python:3.12-slim, pinned. Pin everything: base image, every requirement (fastapi==x.y.z), Postgres image tag (pgvector/pgvector:pg16). Reproducibility is the point.
- Layer caching: order Dockerfile least-changing to most-changing (base -> copy requirements.txt + pip install -> copy source last) so code edits rebuild only the final layer.
- Non-root (least privilege): build steps run as root (installing requires it); then RUN useradd app, chown -R app of the app directory, USER app before the run command. Protects whoever operates the container by boxing in an attacker who achieves code execution inside it; limits damage when something goes wrong.
- Secrets via environment variables: commit env.example with placeholders, gitignore the real .env, compose injects it.
- Volume mounts (host folder mapped into container) required for Postgres data persistence; also used for persistent log files.

## Logging and monitoring

- Structured logs to stdout; Docker captures automatically. Read from host project root: docker compose logs -f api (history + live follow). compose down deletes captured logs; stop/start preserves. For persistence: dump to file (docker compose logs api > snapshot.txt) or log to a volume-mounted directory.
- Log levels: DEBUG < INFO < WARNING < ERROR < CRITICAL. The developer assigns the level per message by choosing which logger function to call. The configured threshold (LOG_LEVEL env var) is a gate at write time: below-threshold calls produce nothing anywhere (not hidden - never recorded). To get debug detail for an issue: restart with LOG_LEVEL=DEBUG and reproduce.
- Configuration is a few lines of logging.basicConfig(level=os.getenv("LOG_LEVEL","INFO"), format=...) in the main module (main.py, the entry point that runs first; uvicorn main:app = "serve the app object in main.py"). Configure once at startup; all modules inherit.
- Sprinkled in as built: phase 1 establishes the configuration; every later component ships with its log lines the day it's written (e.g., retrieval logs chunk count + latency at INFO, per-chunk scores at DEBUG). /health endpoint returns OK if app + DB reachable (what platforms poll).
- Design log lines by asking "what would I want to see while this runs."

## Error handling

- FastAPI global exception handler (the n8n error-flow pattern translated): on any unhandled error, generate short error_id; log full detail with the id; notify developer via Slack/Discord webhook (async); return graceful JSON message to the user including the reference id so specific incidents can be located in logs.
- LLM-interprets-the-error flourish deferred; templated messages are standard practice.

## Ingestion

- CLI script (python ingest.py --dataset scifact) batch-ingests the full corpus: loop over documents, chunk, embed, insert. Expect minutes; log progress by count interval (if i % 100 == 0: logger.info(f"ingested {i}/{total}")) with light telemetry (chunks so far, rate). Skip decisions logged at DEBUG.
- Idempotent and resumable, no queue needed at this scale: document IDs are deterministic (dataset-provided IDs, or content/filename hash), stored in chunk metadata (also links chunks to sources for citations). Before processing a document, SELECT whether chunks with that ID exist; skip if so. The database is the progress record; no separate state.
- Each document's inserts wrapped in one transaction (with db.begin():) - all-or-nothing commit/rollback makes partially stored documents impossible, which is what keeps the existence-check a trustworthy resume marker. True queue/worker system mentioned in README as the scale-up path only.

## Guardrails (four built layers)

Principle: a system prompt instruction is a request; a guardrail is enforcement in code. For each failure mode, ask what code catches it when the prompt fails.

1. Input (before retrieval): query length caps, empty rejection, optional rate limit. Mostly free via Pydantic validation on FastAPI endpoints.
2. No-answer path (after retrieval, before generation): if top similarity scores fail a threshold, return honest "not in my corpus" WITHOUT calling the generator - the hallucination path becomes structurally unreachable. Most important guardrail in a RAG system. EVALS.md experiment: add out-of-domain queries to the golden set, measure refusal rate before/after.
3. Output (after generation): (a) schema-validate all structured LLM output (judge JSON) - retry once then fail loudly, never ingest garbage into metrics; (b) citation-based grounding - require generator to cite chunk IDs, verify cited IDs exist in the retrieved set, strip/flag uncited claims. (Full per-answer faithfulness checks at serve time noted as the expensive strong version.)
4. Resource: timeouts on every LLM call, cap on chunks fed to generation, output token limits. The global exception handler is the backstop under all layers.

Deliberately NOT built: guardrail frameworks (Guardrails AI, NeMo, Llama Guard) - scope creep for a fixed scientific corpus. README "Guardrails" section documents the four layers plus a sentence on what production would add (content moderation for open domains, prompt-injection screening for user-uploaded corpora). Stating the scope boundary is itself production thinking.

Guardrails and evaluation are one discipline from two sides: the no-answer threshold is tuned with the golden set; faithfulness is both guardrail concept and RAGAS metric. Design together. The repo's one-sentence thesis: this system's quality is measured, enforced, and observable.

## Build phasing

1. Core pipeline: ingest -> chunk -> embed -> retrieve -> generate, FastAPI on top. Logging configuration established here; log lines added with each component from now on.
2. Golden set + hand-built retrieval metrics (recall@k, MRR) - BEFORE tuning begins, so every improvement is measured.
3. RAGAS / LLM-as-judge for generation quality.
4. Dockerfile + compose (pinned, non-root, cached layers, volumes).
5. CI with tests and the retrieval eval gate.
- Guardrail layers land where they naturally attach (input validation with the API, no-answer threshold once retrieval metrics exist to tune it, output checks with the judge, resource limits with the LLM calls).
- Each phase independently demo-able. EVALS.md accumulates throughout rather than being written at the end.