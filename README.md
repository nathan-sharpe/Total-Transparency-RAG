# RAG Project

[![CI](https://github.com/nathan-sharpe/RAG_Project/actions/workflows/ci.yml/badge.svg)](https://github.com/nathan-sharpe/RAG_Project/actions/workflows/ci.yml)

A Python-first RAG system over the SciFact corpus: chunk → embed → ingest into
Postgres/pgvector, retrieve + generate via local Ollama models, FastAPI on top,
with two-layer evaluation (hand-built retrieval metrics + LLM-as-judge).

One-sentence thesis: **this system's quality is measured, enforced, and observable.**

- [Constraints_and_context.md](Constraints_and_context.md) — stack decisions and the why behind them.
- [ROADMAP.md](ROADMAP.md) — build phases; each ends demo-able.

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

## Status

Phase 4 complete — see [ROADMAP.md](ROADMAP.md) for the full phase plan and
per-phase implementation notes.

- **Phase 0** — foundations: config, schema, Postgres/pgvector via compose.
- **Phase 1** — core pipeline: ingest → chunk → embed → retrieve → generate, FastAPI on top.
- **Phase 2** — golden set + hand-built retrieval metrics (recall@k, MRR) + guardrail 2 (no-answer threshold).
- **Phase 3** — LLM-as-judge generation eval + guardrails 3a (schema-validated judge output) and 3b (citation grounding).
- **Phase 4** — containerization: API Dockerfile (non-root, pinned base), compose `api` service, log persistence, backstop error webhook.

What's runnable now:

```powershell
docker compose up -d --build         # run the API + db in containers (Ollama native on host)
python ingest.py --dataset scifact   # batch-ingest the corpus
python main.py                       # or serve the API directly on the host at :8000/docs
python evals/run_retrieval.py        # Tier-1 retrieval metrics -> evals/results/retrieval.json
python run_evals.py                  # Tier-2 generation eval (laptop-only) -> evals/results/generation.json
```

Measured baselines and methodology live in [EVALS.md](EVALS.md). Next up:
Phase 5 (CI with the retrieval eval gate) — workflow in place, floor
calibration in progress.
