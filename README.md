# RAG Project

[![CI](https://github.com/nathan-sharpe/RAG_Project/actions/workflows/ci.yml/badge.svg)](https://github.com/nathan-sharpe/RAG_Project/actions/workflows/ci.yml)

A Python-first RAG system over the SciFact corpus: chunk → embed → ingest into
Postgres/pgvector, retrieve + generate via local Ollama models, FastAPI on top,
with two-layer evaluation (hand-built retrieval metrics + LLM-as-judge).

One-sentence thesis: **this system's quality is measured, enforced, and observable.**

- [Constraints_and_context.md](Constraints_and_context.md) — stack decisions and the why behind them.
- [ROADMAP.md](ROADMAP.md) — build phases; each ends demo-able.

## Architecture

```
            ingest.py (CLI, idempotent)                 main.py (FastAPI, thin)
                 │                                            │  /query
   load → chunk → embed → insert                              ▼
                 │                          embed query → pgvector top-k → no-answer gate
                 ▼                                            │ (similarity ≥ threshold?)
        Postgres + pgvector  ◄────────────────────────────────┘
        (chunks + ingestion_meta)                             ▼
                                            Ollama generate (cited answer) → citation
                                            grounding check → JSON response
```

Core logic is plain Python under `rag/` — FastAPI is a thin layer over it, and
the eval scripts import the same functions directly (never over HTTP), so what
gets measured is exactly what gets served:

- `rag/config.py` — every knob as an env var via pydantic-settings; secrets are `SecretStr`.
- `rag/chunking.py` — hand-built word-window chunker; deterministic `{doc_id}::{chunk_index}` IDs.
- `rag/embedding.py` — pluggable embedder interface: Ollama `nomic-embed-text` or CPU
  sentence-transformers, selected by `EMBEDDING_PROFILE`. The ingesting model + dimension is
  recorded in `ingestion_meta`, and query paths refuse a mismatched embedder.
- `rag/retrieval.py` — cosine top-k via pgvector; chunks always return with similarity scores.
- `rag/generation.py` — prompt assembly + Ollama call (timeout, chunk cap, token limit) +
  citation grounding.
- `evals/` — golden set, hand-built metrics (recall@k, MRR), threshold sweep, LLM-as-judge
  (`evals/judge.py`), and the CI gate. Results land in `evals/results/` as JSON with full
  config snapshots.

## Setup

Requires Python 3.12, Docker Desktop, and (from Phase 1) [Ollama](https://ollama.com) running natively on the host.

```powershell
# 1. Virtual environment
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. Configuration — copy the template and fill in real values (never commit .env)
copy env.example .env

# 3. Start Postgres + pgvector
docker compose up -d

# 4. Bootstrap the database schema
python -m rag.db

# 5. Verify everything works
pytest
ruff check .
```

## Run the whole stack in Docker

The API is containerized; Postgres/pgvector runs alongside it via compose.
Ollama stays **native on the host** (the container reaches it at
`host.docker.internal`), so the RTX 4050 serves models directly.

```powershell
copy env.example .env          # set POSTGRES_USER / POSTGRES_PASSWORD
docker compose up -d --build   # start db + api (api image builds on first run)
python ingest.py --dataset scifact   # ingest from the host into the compose db
# open http://localhost:8000/docs and query
```

The `api` service inherits shared config from `.env` and overrides only the
values that differ inside a container (`DB_HOST=db`,
`OLLAMA_URL=http://host.docker.internal:11434`, `API_HOST=0.0.0.0`), so the
same `.env` drives both host-run scripts and the container. The API logs to
`./logs/api.log` on the host (bind-mounted, persists across `compose down`).
If you start the stack before ingesting, `/query` returns `503` until a corpus
exists — no restart needed after `ingest.py` runs.

Optional: set `ERROR_WEBHOOK_URL` in `.env` to a Slack/Discord incoming
webhook and the global exception handler posts a short alert (error id + route,
never the stack trace) on unhandled errors.

## CI

Every push runs [`.github/workflows/ci.yml`](.github/workflows/ci.yml): ruff +
pytest, then the Tier-1 retrieval eval end to end — a Postgres/pgvector service
container, the full SciFact corpus ingested with the CPU embedding profile
(`EMBEDDING_PROFILE=sentence-transformers`, no GPU or Ollama in CI), and a gate
that fails the build if recall@5 drops below `RECALL5_FLOOR`.

**The CI floor is a regression tripwire, not the portfolio number.** It is
calibrated with the CPU profile (all-MiniLM-L6-v2); the numbers that describe
this system's quality live in [EVALS.md](EVALS.md) and come from the Ollama
profile (nomic-embed-text). The two are not comparable, and the gate enforces
that with `--expect-profile`. Tier-2 generation evals (`run_evals.py`) never
run in CI — they need local Ollama models.

CI ingests the **full corpus** (comparability beat the ~10-minute guidance;
a run takes ~12.5 minutes, ingestion dominating). Floor calibrated 2026-07-18:
recall@5 measured **0.7209** under the CPU profile, floor set at **0.68**.

## Guardrails

The design principle: **a system-prompt instruction is a request; a guardrail
is enforcement in code.** For every failure mode the question was "what code
catches this when the prompt fails?" Four layers are built:

1. **Input validation** — Pydantic request model: query length cap
   (`MAX_QUERY_CHARS`), empty-query rejection. Malformed input never reaches
   the pipeline.
2. **No-answer path** — if the best retrieved chunk's cosine similarity is
   below `NO_ANSWER_THRESHOLD`, the API refuses honestly *before* calling the
   generator. The threshold is tuned per embedding profile against measured
   score distributions (see the threshold-sweep entry in
   [EVALS.md](EVALS.md)), trading ~1–2% false refusals for refusing 100% of
   out-of-domain probes.
3. **Output checks** — the judge's JSON is schema-validated (retry once, then
   fail loudly — garbage never enters metrics), and at serve time every cited
   chunk ID is verified against the retrieved set: hallucinated citations are
   stripped and surfaced in an `ungrounded_citations` field.
4. **Resource limits** — every LLM call site has a timeout, a cap on chunks
   fed in, and an output token limit; a global exception handler returns a
   reference-id JSON error (optionally alerting via webhook) instead of a
   stack trace.

Production hardening beyond this scope would add content moderation and
prompt-injection screening in front of layer 1.

## Code-first, or n8n?

This system is deliberately zero-n8n — but the interesting question is *when
each is right*, and the answer comes down to where your complexity lives.

**n8n earns its keep** when the work is integration-shaped: webhook in,
transform, route to CRM/Slack/email, done. The visual graph *is* the
documentation, non-developers can maintain it, and a library of prebuilt nodes
replaces glue code that would otherwise be written and babysat. A RAG
prototype wired this way can be live in an afternoon.

**Code wins** when the complexity is in the logic itself, which is exactly a
RAG system's situation: chunking strategy, retrieval scoring, threshold gates,
prompt assembly, and metrics all need to be *versioned, unit-tested, and
measured*. In this repo, every eval imports the exact functions that serve
traffic; a chunk-size experiment is a git-diffable config change with a
before/after table in EVALS.md; CI replays the whole pipeline on every push.
None of that has an n8n equivalent — a workflow graph can't be
property-tested, its "diff" is a JSON blob, and its eval story is manual
clicking. The rule of thumb: **if the value is in connecting systems, use the
workflow tool; if the value is in the logic between the connections, write
code.** (The error-handling pattern here — catch-all handler, short reference
id, webhook alert — is the n8n error-flow translated into FastAPI.)

## Scaling up

Choices that are right at 5k documents and how they'd change at 5M:

- **Ingestion**: the CLI loop is idempotent and resumable (the database is the
  progress record), which is all a single-writer corpus needs. At scale, the
  same chunk/embed/insert unit moves behind a queue (e.g. Redis/RQ or SQS)
  with N workers; idempotent doc-level transactions mean retries stay safe.
- **Retrieval**: pgvector with no index is exact and fast at 8k chunks; at
  millions, add an HNSW index (approximate, tunable recall) — the query code
  doesn't change.
- **Evals**: the golden set and judge runs are laptop-scale by design;
  scaled-up they become sampled nightly jobs with the same JSON artifacts.

## Status

Phase 6 in progress — see [ROADMAP.md](ROADMAP.md) for the full phase plan and
per-phase implementation notes.

- **Phase 0** — foundations: config, schema, Postgres/pgvector via compose.
- **Phase 1** — core pipeline: ingest → chunk → embed → retrieve → generate, FastAPI on top.
- **Phase 2** — golden set + hand-built retrieval metrics (recall@k, MRR) + guardrail 2 (no-answer threshold).
- **Phase 3** — LLM-as-judge generation eval + guardrails 3a (schema-validated judge output) and 3b (citation grounding).
- **Phase 4** — containerization: API Dockerfile (non-root, pinned base), compose `api` service, log persistence, backstop error webhook.
- **Phase 5** — CI: ruff + pytest + full-corpus retrieval eval against a pgvector service container, gated on recall@5.
- **Phase 6** (in progress) — measured tuning experiments: embedding-model comparison (MiniLM beats nomic by ~17 recall@5 points on SciFact), chunk-size sweep (300/60 wins), per-profile no-answer threshold sweep; generation-side experiments and final doc polish remain.

What's runnable now:

```powershell
docker compose up -d --build         # run the API + db in containers (Ollama native on host)
python ingest.py --dataset scifact   # batch-ingest the corpus
python main.py                       # or serve the API directly on the host at :8000/docs
python evals/run_retrieval.py        # Tier-1 retrieval metrics -> evals/results/retrieval.json
python run_evals.py                  # Tier-2 generation eval (laptop-only) -> evals/results/generation.json
```

Measured baselines, methodology, and every tuning experiment's
change → hypothesis → before/after → conclusion live in [EVALS.md](EVALS.md).
