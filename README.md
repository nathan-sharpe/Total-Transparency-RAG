# RAG Project

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

## Status

Phase 3 complete — see [ROADMAP.md](ROADMAP.md) for the full phase plan and
per-phase implementation notes.

- **Phase 0** — foundations: config, schema, Postgres/pgvector via compose.
- **Phase 1** — core pipeline: ingest → chunk → embed → retrieve → generate, FastAPI on top.
- **Phase 2** — golden set + hand-built retrieval metrics (recall@k, MRR) + guardrail 2 (no-answer threshold).
- **Phase 3** — LLM-as-judge generation eval + guardrails 3a (schema-validated judge output) and 3b (citation grounding).

What's runnable now:

```powershell
python ingest.py --dataset scifact   # batch-ingest the corpus
python main.py                       # serve the API at http://localhost:8000/docs
python evals/run_retrieval.py        # Tier-1 retrieval metrics -> evals/results/retrieval.json
python run_evals.py                  # Tier-2 generation eval (laptop-only) -> evals/results/generation.json
```

Measured baselines and methodology live in [EVALS.md](EVALS.md). Next up:
Phase 4 (containerization).
