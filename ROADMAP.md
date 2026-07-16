# ROADMAP

Companion to [Constraints_and_context.md](Constraints_and_context.md) — that file holds the *what and why*; this file holds the *when and in what order*. Phases 0–2 are specified in detail; later phases are directional and get elaborated as they approach. Each phase ends in something independently demo-able.

**One-sentence thesis (from the constraints doc):** this system's quality is measured, enforced, and observable.

---

## Design decisions locked in early (to avoid backtracking)

These choices are made in Phase 0/1 specifically because a later phase depends on them. When implementing early phases, treat these as requirements, not suggestions.

| Decision | Made in | Pays off in |
|---|---|---|
| Pluggable embedder interface (Ollama `nomic-embed-text` primary, small sentence-transformers model for CI); embedding model name + dimension recorded in the DB | Phase 1 | Phase 5 (CI runs the same pipeline with the CPU profile), Phase 6 (embedding-model comparison experiments) |
| Deterministic chunk IDs of the form `{doc_id}::{chunk_index}`; source doc ID kept in chunk metadata | Phase 1 | Phase 2 (mapping SciFact's document-level labels to chunk-level credit), Phase 3 (citation grounding), ingestion idempotency |
| Retrieval returns similarity scores, not just chunks | Phase 1 | Phase 2 no-answer threshold (guardrail 2), DEBUG logging |
| Core logic (chunking, embedding, retrieval, generation) lives in plain Python modules; FastAPI is a thin layer on top | Phase 1 | Phases 2/3/5: eval scripts and CI import and call the pipeline directly, no HTTP server needed |
| All configuration via environment variables through one settings module from day one | Phase 0 | Phase 4: containerizing means changing env values (`DB_HOST`, `OLLAMA_URL=http://host.docker.internal:11434`), not code |
| Golden-set labels stored as a *list* of relevant IDs per query, even when singleton | Phase 2 | Multi-chunk relevance and nDCG need no schema change |
| Every LLM call takes a timeout parameter; generation input capped in chunks, output capped in tokens | Phase 1 | Guardrail 4 exists structurally from the first call site |
| pytest layout (`tests/`) established with the first module | Phase 1 | Phase 5 CI is wiring, not writing |
| `docker-compose.yml` exists from Phase 0 (Postgres only) and is *extended* in Phase 4 | Phase 0 | No "now dockerize everything" cliff; the compose file grows a service |

---

## Phase 0 — Foundations and scaffolding

**Goal:** a repo where every later phase has a slot to land in. No RAG logic yet.

**Deliverables**

1. **Repo hygiene** (order matters — per global security rules, `.gitignore` exists before any secret does):
   - `.gitignore` covering `.env`, `.env.*`, `__pycache__/`, `.venv/`, data downloads.
   - `.env.example` with placeholder values for every setting (DB connection, `OLLAMA_URL`, `LOG_LEVEL`, `EMBEDDING_PROFILE`); real `.env` never committed.
2. **Python project skeleton:**
   - `requirements.txt` with every dependency pinned (`fastapi==x.y.z` style). Virtual environment documented in README stub.
   - Package layout:
     ```
     rag/
       __init__.py
       config.py        # pydantic-settings: reads env, single source of truth
       db.py            # connection + schema management
       chunking.py      # (Phase 1)
       embedding.py     # (Phase 1) interface + implementations
       retrieval.py     # (Phase 1)
       generation.py    # (Phase 1)
       datasets.py      # (Phase 1) SciFact loader
     ingest.py          # CLI entry point (Phase 1)
     main.py            # FastAPI app entry point (Phase 1)
     evals/             # (Phases 2-3)
     tests/
     ```
   - `ruff` configured (it lints in CI later; run it locally from the start).
3. **Logging configuration** in the entry points: `logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), ...)`, structured single-line format to stdout. Every subsequent component ships with its log lines the day it's written — this phase creates the convention.
4. **Postgres + pgvector via compose:** `docker-compose.yml` with one service — `pgvector/pgvector:pg16` (pinned tag), volume mount for data persistence, credentials from `.env`. Schema bootstrap (`db.py`): a `chunks` table (id, doc_id, chunk_index, text, embedding vector, metadata JSONB) plus a small `ingestion_meta` table recording which embedding model/dimension the corpus was embedded with — queried at startup to refuse mismatched query-time embedders.
5. **Smoke test:** a first pytest that connects to the DB, inserts a dummy vector, and retrieves it by similarity. Proves the whole substrate works.

**Demo:** `docker compose up -d`, `pytest` passes green.

**Forward hooks:** the vector column dimension is set from the active embedder profile at schema-creation time, not hardcoded — this is what makes the CI embedding profile (Phase 5) a config swap.

### Phase 0 — Implementation notes (completed 2026-07-15, commit `aa54782`)

*Convention established with this phase: when a phase completes, a section like this one is appended to it — what was actually built, decisions made during implementation, and anything later phases should know.*

- **Environment:** Python 3.12.10 was newly installed on the Windows host (via winget) as part of this phase. Dependencies pinned in `requirements.txt`: `pydantic-settings`, `psycopg[binary]` (psycopg **3**, not psycopg2), the `pgvector` Python adapter, `pytest`, `ruff`.
- **Database access** uses psycopg 3 with `pgvector.psycopg.register_vector`. `rag.db.connect()` runs `CREATE EXTENSION IF NOT EXISTS vector` before registering the type, so a fresh compose-created database self-initializes on first connection — no manual bootstrap step. Schema creation is runnable directly via `python -m rag.db`.
- **Schema details** beyond the plan:
  - `ingestion_meta` is a singleton row (`id smallint PRIMARY KEY DEFAULT 1 CHECK (id = 1)`) — one corpus per database, by construction.
  - `chunks` has an index on `doc_id`, because Phase 1's ingestion idempotency check ("do chunks for this doc exist?") is a `doc_id` lookup.
  - A pgvector column dimension cannot be a bind parameter, so `vector({dim})` is interpolated into the DDL from validated integer settings — the one sanctioned f-string in a SQL statement.
- **Landed earlier than planned:** the embedder-mismatch refusal (ROADMAP said "queried at startup" — i.e. Phase 1) already exists in `ensure_schema()`: it raises if `ingestion_meta`'s recorded model/dimension, or the existing `chunks` column dimension (recovered from `pg_attribute.atttypmod`), disagrees with the active profile. Phase 1 still needs to invoke this on the query path, but the check itself is written and tested.
- **Config shape:** `EMBEDDING_PROFILES` (profile → model + dimension) lives in `rag/config.py`; `EMBEDDING_PROFILE` selects one, and optional `EMBEDDING_MODEL`/`EMBEDDING_DIMENSION` env vars override it for Phase 6 experiments. `get_settings()` is `lru_cache`d — tests that need different settings construct `Settings` explicitly rather than mutating the cache.
- **Verified:** `docker compose up -d` + `pytest` green, including the smoke test (insert dummy vector, retrieve by similarity) against the real compose database.

---

## Phase 1 — Core pipeline: ingest → chunk → embed → retrieve → generate

**Goal:** ask a question over SciFact and get a grounded answer, end to end.

**Deliverables**

1. **Dataset loader** (`rag/datasets.py`): downloads/loads SciFact (BEIR format), yields `(doc_id, title, text)`. Keep the interface dataset-agnostic — NFCorpus later should mean a new loader function, nothing else.
2. **Chunker** (`rag/chunking.py`), hand-built (no LangChain): configurable chunk size and overlap via settings. Emits chunks carrying `doc_id`, `chunk_index`, deterministic `chunk_id = f"{doc_id}::{chunk_index}"`. SciFact abstracts are short — many will be single-chunk — but the machinery must be real because chunk-size sweeps are a Phase 6 experiment.
3. **Embedder interface** (`rag/embedding.py`): one small abstract interface — `embed_documents(texts) -> list[vector]`, `embed_query(text) -> vector`, plus `model_name` and `dimension` properties. Two implementations:
   - `OllamaEmbedder` (nomic-embed-text) — the primary, local-AI path.
   - `SentenceTransformersEmbedder` (e.g. `all-MiniLM-L6-v2`) — CPU-friendly, used by CI.
   - `EMBEDDING_PROFILE` env var selects one; the chosen model/dimension is what gets recorded in `ingestion_meta`.
4. **Ingestion CLI** (`python ingest.py --dataset scifact [--limit N]`):
   - Idempotent and resumable: before processing a document, check whether chunks with its `doc_id` exist; skip if so (logged at DEBUG). The database *is* the progress record.
   - Each document's chunks inserted in one transaction — all-or-nothing keeps the existence check a trustworthy resume marker.
   - Progress at INFO every 100 docs with light telemetry (count, chunks so far, rate).
   - `--limit` exists from day one: fast local iteration now, CI subset later.
5. **Retrieval** (`rag/retrieval.py`): embed query, cosine similarity search in pgvector, return top-k chunks **with scores**. Logs chunk count + latency at INFO, per-chunk scores at DEBUG.
6. **Generation** (`rag/generation.py`): Ollama chat call (Llama 3.1 8B or Qwen 2.5 7B) with a prompt template that interpolates retrieved chunks *labeled by chunk ID* and instructs the model to cite the IDs it used. Citation *verification* is a Phase 3 guardrail, but the answer format must ask for citations from the first prompt — retrofitting prompt formats invalidates earlier eval numbers. Resource guardrails at every call site: request timeout, cap on chunks fed in, output token limit (guardrail 4).
7. **FastAPI app** (`main.py`):
   - `POST /query` — Pydantic request model enforcing query length caps and empty-string rejection (guardrail 1, mostly free). Response includes the answer, cited chunk IDs, and retrieved sources.
   - `GET /health` — OK if app + DB reachable.
   - Global exception handler: generate short `error_id`, log full detail with it, return graceful JSON containing the id. (Webhook notification deferred to Phase 4 — the handler structure lands now.)
8. **Unit tests** for chunker (deterministic IDs, overlap behavior) and the Pydantic validation.

**Demo:** `docker compose up -d` (DB), `python ingest.py --dataset scifact`, open `http://localhost:8000/docs`, ask a question in the Swagger UI, get an answer with citations.

**Forward hooks:** retrieval and generation callable as plain functions (eval scripts import them); scores returned (threshold tuning); citations requested (grounding checks); `--limit` (CI).

---

## Phase 2 — Golden set and hand-built retrieval metrics

**Goal:** every future change can be measured. This lands **before any tuning begins**.

**Deliverables**

1. **Golden set loader** (`evals/golden.py`): SciFact's ~300 labeled test queries as `query -> [relevant_doc_ids]` (always a list). Because labels are document-level and retrieval is chunk-level, define the credit rule explicitly: a retrieved chunk counts as relevant if its `doc_id` is in the golden list. Document this rule in EVALS.md — it's a methodological choice reviewers will look for.
2. **Metrics by hand** (`evals/metrics.py`, ~50 lines, unit-tested against tiny hand-computed fixtures):
   - `recall_at_k` — ceiling on whole-system quality.
   - `mrr` — position sensitivity (LLMs attend more to earlier context).
   - nDCG: stretch goal, noted but not built.
3. **Retrieval eval runner** (`evals/run_retrieval.py`): loops the golden queries through `rag/retrieval.py` (direct import, no HTTP), prints and writes a JSON results file (metrics + config snapshot: embedder, chunk params, k). The JSON output is what the CI gate (Phase 5) will parse.
4. **EVALS.md created:** corpus and golden-set description, credit rule, baseline recall@5 / recall@10 / MRR with the exact config that produced them. From here on it accumulates dated changelog entries: *change → hypothesis → numbers before/after → conclusion*.
5. **Guardrail 2 — the no-answer path** (lands here because the metrics needed to tune it now exist):
   - In the query flow: if top similarity scores fail a threshold, return an honest "not in my corpus" response *without calling the generator*.
   - Add a small set of out-of-domain queries to the golden set; measure refusal rate before/after as the first EVALS.md changelog entry.

**Demo:** `python evals/run_retrieval.py` prints baseline numbers; EVALS.md shows them with methodology.

---

## Phase 3 — Generation evaluation: LLM-as-judge

**Goal:** the second eval layer — given the chunks, was the answer good?

**Scope (directional):**

- Judge loop: rubric in system prompt; query/chunks/answer interpolated; output constrained to a JSON schema; called per test case against local Ollama; verdicts parsed and aggregated. Evaluate RAGAS's ready-made metrics (faithfulness, context precision, answer relevance) vs. a hand-rolled loop — decide when we get here; the constraints doc supports either.
- **Guardrail 3a lands here:** schema-validate all judge output — retry once, then fail loudly. Garbage never enters metrics.
- **Guardrail 3b lands here:** citation grounding at serve time — verify cited chunk IDs exist in the retrieved set; strip/flag uncited claims. (Full per-answer faithfulness at serve time documented as the expensive strong version, not built.)
- `run_evals.py` — the Tier-2 manual eval (laptop + Ollama, never CI); results committed to EVALS.md by hand.
- Document the 8B-judge noise limitation honestly in EVALS.md; optional one-time re-score with a cheap API model for comparison (key via `.env` only).

**Demo:** `python run_evals.py` produces generation-quality scores; EVALS.md gains a generation baseline section.

---

## Phase 4 — Containerization

**Goal:** "clone, `docker compose up`, running locally."

**Scope (directional):**

- Dockerfile for the API: pinned `python:3.12-slim`, layers ordered least- to most-changing (base → requirements + pip install → source), non-root (`useradd app`, `chown`, `USER app` after root-only build steps).
- Extend the existing compose file: `api` service joins `db`; env-driven config flips `OLLAMA_URL` to `http://host.docker.internal:11434` (Ollama stays native on the Windows host). Volume-mounted log directory for persistence beyond `compose down`.
- Global exception handler gains the async Slack/Discord webhook notification (URL from `.env`).
- Because config has been env-driven since Phase 0, this phase should be packaging work, not refactoring. If it isn't, that's a signal an earlier phase leaked config into code.

**Demo:** fresh clone on a machine with Docker + Ollama: `cp .env.example .env`, `docker compose up`, ingest, query via `/docs`.

---

## Phase 5 — CI with tests and the retrieval eval gate

**Goal:** every push proves the repo still works and retrieval hasn't regressed.

**Scope (directional):**

- `.github/workflows/ci.yml`: checkout → setup Python → install pinned deps → ruff → pytest.
- **Tier-1 retrieval eval in CI:** Postgres+pgvector as a GitHub Actions service container; ingest with the sentence-transformers profile (this is where the pluggable embedder pays off); run `evals/run_retrieval.py`; a gate script exits nonzero if recall@5 drops below a floor.
- Open question to resolve at implementation time: full 5.2K-doc corpus in CI vs. a `--limit` subset (faster, but the floor must be calibrated on the subset, and subset numbers aren't comparable to EVALS.md numbers). Start with whichever keeps CI under ~10 minutes; document the choice.
- CI floor numbers are calibrated with the CI embedding profile — they are *regression tripwires*, not the portfolio numbers (those live in EVALS.md from the Ollama profile). Keep this distinction explicit in the README.
- Self-hosted runner / hosted-API judge in CI: documented stretch goals only.

**Demo:** a push with a deliberately broken retrieval change goes red; the fix goes green.

---

## Phase 6 — Tuning experiments and documentation polish

**Goal:** EVALS.md becomes the portfolio centerpiece it's meant to be; README tells the full story.

**Planned experiments** (each one dated EVALS.md entry: change → hypothesis → before/after → conclusion):

1. Chunk size / overlap sweep (re-ingest per config; `ingestion_meta` keeps runs honest).
2. Top-k sensitivity: recall@k vs. k, and downstream effect on faithfulness (more chunks ≠ better answers).
3. No-answer threshold refinement against the out-of-domain query set (refusal rate vs. false-refusal rate).
4. Prompt variants for the generator, measured by judge faithfulness.
5. Embedding model comparison (nomic-embed-text vs. sentence-transformers profile) — the interface makes this nearly free.
6. Stretch: hybrid retrieval (BM25 + vector), reranking, nDCG.

**Documentation deliverables:**

- README: architecture overview; the **code-first vs. n8n** comparison section (when each is right); the **Guardrails** section documenting the four built layers plus one sentence on what production would add (content moderation, prompt-injection screening); scale-up notes (queue/worker ingestion path).
- EVALS.md final pass for narrative coherence.

---

## Future extensions (explicitly out of scope, listed to show the boundary was chosen)

- Multi-dataset verification (NFCorpus, broader BEIR suite).
- Custom golden-set generation (RAGAS synthesis + manual verification) for domain-specific corpora.
- Self-hosted GitHub runner or hosted-API judge for Tier-2 evals in CI.
- Queue/worker ingestion at scale.
- Guardrail frameworks (Guardrails AI, NeMo, Llama Guard) — scope creep for a fixed scientific corpus.

---

## Guardrail landing map (cross-reference)

| Layer | What | Lands in |
|---|---|---|
| 1. Input | Query length caps, empty rejection (Pydantic) | Phase 1 (with the API) |
| 2. No-answer path | Similarity threshold gate before generation | Phase 2 (once metrics exist to tune it) |
| 3a. Output: schema | Validate judge JSON, retry once, fail loudly | Phase 3 (with the judge) |
| 3b. Output: grounding | Verify cited chunk IDs against retrieved set | Phase 3 |
| 4. Resource | LLM timeouts, chunk cap, token limits | Phase 1 (with the first LLM calls) |
| Backstop | Global exception handler (+ webhook in Phase 4) | Phase 1 |
