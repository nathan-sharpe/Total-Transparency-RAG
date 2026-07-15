# CLAUDE.md

Python-first RAG system over the SciFact corpus: chunk → embed → ingest into Postgres/pgvector, retrieve + generate via local Ollama models, FastAPI on top, with rigorous two-layer evaluation (retrieval metrics + LLM-as-judge).

## Authoritative documents

- **[Constraints_and_context.md](Constraints_and_context.md)** — the *what and why*: stack decisions, evaluation methodology, guardrail design, container/CI requirements. Consult it before designing any component.
- **[ROADMAP.md](ROADMAP.md)** — the *when and in what order*: 7 phases (0–6), each independently demo-able. Follow the phase order; its "Design decisions locked in early" table is binding — those choices exist so later phases don't force rework.
- **EVALS.md** (created in Phase 2) — living eval changelog: change → hypothesis → before/after numbers → conclusion. Update it with every measured change; never retro-write it at the end.

## Hard rules

- **Zero n8n.** Everything runs through code.
- **Hand-build core logic** — chunking, retrieval scoring, recall@k/MRR metrics. No LangChain for these; that's the point of the project.
- **Pin everything**: every requirement (`fastapi==x.y.z`), Docker base images, Postgres image tag (`pgvector/pgvector:pg16`).
- **Config only via env vars** through `rag/config.py` (pydantic-settings). Never hardcode hosts, model names, thresholds, or credentials in module code.
- **Prompt formats are eval-sensitive.** Changing the generator or judge prompt invalidates prior EVALS.md numbers — treat prompt changes as measured experiments, not casual edits.

## Architecture conventions

- Core logic lives in plain Python modules under `rag/`; FastAPI (`main.py`) is a thin layer. Eval scripts import `rag/` functions directly — never call the pipeline over HTTP internally.
- Embedding goes through the pluggable embedder interface (`rag/embedding.py`): `OllamaEmbedder` (nomic-embed-text, primary) and `SentenceTransformersEmbedder` (CPU, used by CI), selected by `EMBEDDING_PROFILE`. The active model name + dimension is recorded in `ingestion_meta`; refuse query-time embedders that mismatch the ingested corpus.
- Chunk IDs are deterministic: `{doc_id}::{chunk_index}`. Chunk metadata always carries the source `doc_id` (needed for golden-set credit mapping and citation grounding).
- Retrieval functions return chunks **with similarity scores**.
- Golden-set labels are always a **list** of relevant IDs per query, even when singleton.
- Every LLM call site has a timeout, a cap on chunks fed in, and an output token limit.
- Ingestion is idempotent: check for existing `doc_id` chunks before processing; wrap each document's inserts in one transaction. The database is the progress record.

## Logging

- Configured once in entry points via `logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), ...)`, structured single-line format to stdout.
- Every new component ships with its log lines the day it's written. Convention: operational milestones at INFO (chunk counts, latency, ingestion progress every 100 docs), per-item detail at DEBUG (per-chunk scores, skip decisions).
- Design log lines by asking "what would I want to see while this runs."

## Guardrails principle

A system-prompt instruction is a request; a guardrail is enforcement in code. For each failure mode, ask what code catches it when the prompt fails. The four layers (input validation, no-answer threshold, output schema/citation checks, resource limits) land in the phases mapped at the bottom of ROADMAP.md — don't defer one past its mapped phase.

## Environment

- Windows 11 host; Ollama runs **natively** on the host (`http://localhost:11434` locally; containers reach it at `http://host.docker.internal:11434`).
- Hardware budget: RTX 4050 6GB VRAM — 7–8B quantized models (Llama 3.1 8B / Qwen 2.5 7B) and nomic-embed-text only. Don't suggest larger local models.
- Postgres + pgvector runs in Docker via `docker compose up -d` with a volume mount for persistence.

## Commands (as they come online per phase)

- `docker compose up -d` — start Postgres/pgvector (later: + API).
- `python ingest.py --dataset scifact [--limit N]` — batch ingestion CLI.
- `pytest` — unit tests; `ruff check .` — lint. Both must pass before any commit (CI runs them on every push from Phase 5).
- `python evals/run_retrieval.py` — retrieval metrics (Phase 2+); `python run_evals.py` — Tier-2 generation evals, laptop-only, never in CI (Phase 3+).

## Testing

- Tests live in `tests/`, written alongside each module from Phase 0 onward (CI in Phase 5 is wiring, not writing).
- Metric implementations are unit-tested against tiny hand-computed fixtures.
- Judge/eval output is schema-validated: retry once, then fail loudly — garbage never enters metrics.
