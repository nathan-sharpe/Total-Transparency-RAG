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
   - `env.example` with placeholder values for every setting (DB connection, `OLLAMA_URL`, `LOG_LEVEL`, `EMBEDDING_PROFILE`); real `.env` never committed. (Named without a leading dot so it sits outside the `.env*` ignore/deny rules — see the Phase 1 notes.)
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

### Phase 1 — Implementation notes (completed 2026-07-15)

Built in the locked order (loader → chunker → embedder → ingest → retrieval → generation → API). All 36 unit tests pass, ruff clean, and the whole pipeline was exercised against the real corpus and live local models.

**New pinned dependencies:** `httpx==0.28.1` (Ollama client + dataset download), `fastapi==0.139.0`, `uvicorn==0.51.0`. `sentence-transformers==5.6.0` (drags in `torch==2.13.0`) is split into a separate **`requirements-ci.txt`** (`-r requirements.txt` + the one extra line) rather than the base file — the Ollama path never needs torch, and CI (Phase 5) is the only consumer of the CPU profile. Installing it locally is only needed to test that profile.

**Loader (`rag/datasets.py`):** SciFact is **5,183 documents**. Downloads BEIR's zip via a `.part` temp file so an interrupted download never looks complete; `corpus.jsonl` existing is the idempotency marker. `Document(doc_id, title, text)` NamedTuple; `DATASETS` registry maps CLI names to loaders (NFCorpus later = one function + one entry). `_id` is coerced to `str` because chunk IDs are strings.
- **Gotcha (carried forward from the eventual Phase 4/CI notes):** stdlib `urllib` fails TLS on this Windows host (`CERTIFICATE_VERIFY_FAILED` — Windows Python doesn't use certifi's CA bundle). Switched the download to `httpx`, which verifies against certifi automatically. Anything else reaching the network from Python on this host should use `httpx`, not `urllib`.

**Chunker (`rag/chunking.py`):** sizes measured in **words, not tokens** (no tokenizer dependency, deterministic, ~1.3 tokens/word). Defaults `CHUNK_SIZE=200`, `CHUNK_OVERLAP=40`. Ingestion feeds it `f"{title}\n\n{text}"` so titles are searchable — that composition lives in `ingest.py`, not the chunker. Full corpus → **8,184 chunks** (corrected during Phase 5 — 8,104 was a transcription error; both the local `retrieval.json` snapshot and CI measure 8,184) (~48% single-chunk, ~47% two-chunk, a long tail up to 10) — the overlap machinery genuinely exercises despite short abstracts.

**Embedder (`rag/embedding.py`):** `embed_documents` / `embed_query` are split because nomic-embed-text is asymmetric — it needs the `search_document: ` / `search_query: ` task prefixes applied at embed time (Ollama does not add them; omitting them hurts retrieval). Those prefixes are nomic-specific, flagged for revisit if `EMBEDDING_MODEL` is overridden. `OllamaEmbedder` uses the batch `/api/embed` endpoint and validates returned dimension against the profile. `OllamaEmbedder.__init__` accepts an injectable `httpx` transport purely so unit tests mock it with `httpx.MockTransport` (no server).

**Ingestion (`ingest.py`):** connection is `autocommit=True` with an explicit `conn.transaction()` per document, so the resume SELECTs don't hold a transaction open between docs while each document's chunks stay all-or-nothing. `ingestion_meta` is written once via `INSERT ... ON CONFLICT (id) DO NOTHING`. `httpx`'s per-request INFO logging is silenced to WARNING (thousands of embed calls). **Throughput: ~10.3 docs/s, full corpus in ~504s (~8.5 min)** — embedding-bound on the RTX 4050. `--limit` re-run confirmed idempotency (50 docs → all skipped, 0 written).

**Retrieval (`rag/retrieval.py`):** cosine via pgvector `<=>` (distance), similarity reported as `1 - distance`. `verify_corpus_compatible()` is the query-path half of the mismatch guard Phase 0 built into `ensure_schema` — it's called at API startup (fail fast), not per query. First query pays a ~2s cold-embed cost; warm queries are sub-second.

**Generation (`rag/generation.py`):** prompt is **eval-sensitive** — labels each chunk by ID, asks for `[chunk_id]` citations, and mandates an exact refusal string. This format is frozen as the baseline; changing it later is a measured experiment, not an edit. Guardrail 4 (timeout / chunk cap / token limit) is enforced here from settings. Citation extraction is a regex over `{doc_id}::{index}`, deduped in first-mention order. Verification of citations is deferred to Phase 3 as planned.

**API (`main.py`):** thin. Guardrail 1 (empty rejection + length cap) is a Pydantic `field_validator` that reads the limit from settings at request time. Embedder is built once in the `lifespan` startup, which also runs `verify_corpus_compatible`. Global exception handler returns `{error, error_id}` and logs the trace under that id.

**Live end-to-end sanity checks (not automated — they need Ollama + an ingested corpus):**
- "gut microbial diversity vs. arterial stiffness" → top chunk `13714201::0` at **0.910**, grounded answer citing `[13714201::0]`. Verified identically through `POST /query`.
- A vaguer query ("does a high-fat diet affect the gut microbiome?") retrieved related-but-not-specific chunks (top 0.78) and the model returned the exact refusal string. Encouraging for the Phase 2 no-answer threshold — refusal behavior already emerges from the prompt, and the score gap (0.91 answerable vs. 0.78 refused) is roughly where guardrail 2's threshold will live.

**Deferred / for later phases:** generation quality is unmeasured until the Phase 2 golden set and Phase 3 judge exist — the two anecdotes above are smoke tests, not evidence. `all-MiniLM-L6-v2` (CI profile) is installed and dimension-checked but not yet run through ingestion end to end; that happens in Phase 5.

**Config template renamed `.env.example` → `env.example`:** the developer's global Claude Code settings deny `Read`/`Write` on `.env.*`, which caught the placeholder template as collateral and made it unmaintainable by the assistant. Dropping the leading dot moves it outside the `.env*` ignore/deny globs while every real secret file (`.env`, `.env.local`, …) stays blocked — no settings change, no loss of the wildcard safety net. `.gitignore`, the README setup step, and the Phase 4 demo command were updated to match. The template now also documents Phase 1's optional overrides (chunk size, top-k, generator model, guardrail caps) as commented defaults.

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

### Phase 2 — Implementation notes (completed 2026-07-16)

Built the eval harness under `evals/` (new package, imports `rag/` directly — never HTTP) plus guardrail 2. All 56 unit tests pass (20 new), ruff clean.

**Golden set (`evals/golden.py`):** the test split is the join of `qrels/test.tsv` (query-id → relevant corpus-ids, score-filtered) with `queries.jsonl` (id → text) — **300 queries**. `queries.jsonl` holds all 1,109 train+test queries; a query belongs to a split iff it appears in that split's qrels, so the join *is* the split filter. `GoldenQuery` keeps `relevant_doc_ids` as a list even when singleton (per the locked design decision). Results are sorted numeric-ascending by query id for reproducibility. Also carries `OUT_OF_DOMAIN_QUERIES` — 8 hand-authored off-topic queries (no relevant doc) for the no-answer measurement.

**Metrics (`evals/metrics.py`):** pure functions over `(ranked_ids, relevant_ids)` — `recall_at_k`, `reciprocal_rank`, plus `mean_recall_at_k` / `mrr` aggregators. The **credit rule** (chunk relevant iff its `doc_id` ∈ golden list) lives in the *runner*, which feeds each chunk's `doc_id` in rank order into these generic functions; keeping the math ID-agnostic is what lets it be unit-tested against tiny hand-computed fixtures. **`k` is counted in chunks, not documents** — documented in EVALS.md as the reviewer-facing methodological choice.

**Runner (`evals/run_retrieval.py`):** retrieves `RETRIEVE_K=10` chunks per query (covers both recall@5 and recall@10 in one call), writes `evals/results/retrieval.json` (metrics + full config snapshot — the Phase 5 CI gate parses this) and prints a summary. Silences `rag.retrieval`'s per-query INFO line to WARNING for the run. A `sys.path` bootstrap supports both `python evals/run_retrieval.py` (the roadmap demo command, which otherwise puts `evals/` rather than the repo root on the path) and `python -m evals.run_retrieval`.

**Baseline (Ollama profile, chunk 200/40):** recall@5 **0.5535**, recall@10 **0.6024**, MRR **0.4671** over 300 queries in ~31s. The recall@5→@10 gap is only ~5 points, so most gettable docs are already in the top 5.

**Guardrail 2 (no-answer path):** `no_answer_threshold` added to config (initial **0.60**); `is_answerable(chunks, threshold)` — a pure function in `rag/retrieval.py` — is wired into `main.py`'s `/query` so a below-threshold top score returns the canonical refusal **without calling the generator** (sources still returned). The refusal string was extracted into `NO_ANSWER_RESPONSE` in `generation.py` and reused both in the (frozen, eval-sensitive) system prompt and by the guardrail, so both layers refuse in identical wording — a lock test asserts the exact string stays in the prompt. **Measured first changelog entry in EVALS.md:** at 0.60 the gate refuses 6/8 out-of-domain queries with **0/300** in-domain false refusals. All 300 in-domain queries cleared 0.60, so the threshold has headroom — tuning it to catch the last 2 OOD queries without inducing false refusals is deferred to the Phase 6 threshold sweep.

**Deferred as planned:** nDCG (noted, not built); threshold tuning (Phase 6); the Phase 5 CI gate will parse the JSON this phase produces.

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

### Phase 3 — Implementation notes (completed 2026-07-17)

Built the second eval tier: a hand-rolled LLM-as-judge plus guardrails 3a/3b. Decision on RAGAS vs. hand-rolled: **hand-rolled**, consistent with the project's "hand-build core logic" rule — the loop is rubric prompt + JSON-schema validation + aggregation, about the same size as learning RAGAS's abstractions and it keeps the eval story fully owned. All 67 unit tests pass (11 new), ruff clean.

**Judge (`evals/judge.py`):** scores **faithfulness** and **relevance** (1–5) on a frozen, eval-sensitive rubric. Context *precision* deliberately omitted — Tier 1 already measures retrieval against real labels, stronger than an 8B model guessing. `JudgeVerdict` (pydantic) declares `reasoning` **first** so the constrained decoder reasons before scoring. **Guardrail 3a** is two-layer: Ollama's `format=` constrained decoding enforces JSON shape, and pydantic validation enforces bounds (`1 ≤ score ≤ 5`) the schema can't; on failure it retries once with the validation error fed back into the message list (temperature is 0, so a naive retry would just repeat the bad output), then raises `JudgeError`.

**Judge model:** `qwen2.5:7b`, pulled this phase — deliberately **different** from the generator (`llama3.1:8b`) so no model scores its own output. Config gained `judge_model` + temp/timeout/token guardrails (guardrail 4 at this call site too).

**Guardrail 3b (`ground_citations` in `rag/generation.py`):** verifies cited chunk IDs against the retrieved set, strips hallucinated IDs from the answer text (tidying the punctuation/spacing a removed `[id]` leaves behind), and reports them separately. Wired into `main.py`'s `/query`, which gained an `ungrounded_citations` response field. Unit-tested against hallucinated-ID fixtures — it must not depend on the corpus to exercise it.

**Runner (`run_evals.py`):** Tier-2, laptop-only, never CI. **Two passes** (generate all → judge all) because 6 GB VRAM fits one 7–8B model at a time; interleaving would thrash models in/out of VRAM per query. Pass 1 writes every answer to `evals/results/answers.json` (gitignored — bulky, holds chunk texts) *before* judging, so a judging failure never loses the ~52-min generation work; a `--resume-judge` flag judges those saved answers without regenerating. Committed artifact is `evals/results/generation.json` (like `retrieval.json`). Refusals are counted, not judged.

**Baseline (full 300-query test split, ~52 min gen + ~21 min judge):** 109 answered, **191 refused by the generator**, 0 by guardrail 2, 0 judge failures, 0 ungrounded citations. Answered: faithfulness **3.93/5**, relevance **4.33/5**, faithfulness≥4 **67%**. The load-bearing finding — documented in EVALS.md — is that the generator refuses 64% of queries *despite* every one clearing the retrieval threshold (relevant chunks were retrieved), because SciFact "queries" are declarative **claims**, not questions. Tier 1 says retrieval works ~55%; the end-to-end system answers 36%; the gap is a generation/prompt problem, and reframing the claim in the generator prompt is flagged as the highest-leverage Phase 6 experiment.

**Deferred as planned:** hosted-frontier-model re-score to calibrate 8B-judge noise (optional stretch, key via `.env` only — noted honestly in EVALS.md); full per-claim serve-time faithfulness (the expensive strong version of 3b); prompt-variant experiment to attack the refusal rate (Phase 6, run as a measured before/after).

---

## Phase 4 — Containerization

**Goal:** "clone, `docker compose up`, running locally."

**Scope (directional):**

- Dockerfile for the API: pinned `python:3.12-slim`, layers ordered least- to most-changing (base → requirements + pip install → source), non-root (`useradd app`, `chown`, `USER app` after root-only build steps).
- Extend the existing compose file: `api` service joins `db`; env-driven config flips `OLLAMA_URL` to `http://host.docker.internal:11434` (Ollama stays native on the Windows host). Volume-mounted log directory for persistence beyond `compose down`.
- Global exception handler gains the async Slack/Discord webhook notification (URL from `.env`).
- Because config has been env-driven since Phase 0, this phase should be packaging work, not refactoring. If it isn't, that's a signal an earlier phase leaked config into code.

**Demo:** fresh clone on a machine with Docker + Ollama: `cp env.example .env`, `docker compose up`, ingest, query via `/docs`.

### Phase 4 — Implementation notes (completed 2026-07-18)

As predicted, this was packaging work, not refactoring — no module code changed to make config container-ready; the container-specific values are supplied by compose. All 71 unit tests pass (4 new), ruff clean, and the full stack was exercised end to end through the container.

**Dockerfile:** pinned `python:3.12.10-slim` (full patch pin, matching the host interpreter). Layers ordered least- to most-changing — `COPY requirements.txt` + `pip install` sit above `COPY . .`, so a source edit reuses the cached dependency layer (rebuild after a code-only change is ~3s). Non-root: `useradd app` runs after the root-only pip install, then `COPY --chown=app:app` and `USER app`. `CMD ["python", "main.py"]`, which reads `API_HOST`/`API_PORT` from the environment — no host/port baked into the image. `PYTHONUNBUFFERED=1` so container logs stream live. The image installs `requirements.txt` only (not `requirements-ci.txt`) — the API serves the Ollama profile and never needs torch; keeping torch out holds the image small.

**`.dockerignore` added:** without it the 500 MB+ `.venv/` and the `data/` corpus download would land in the build context. It also blocks `.env`/`*.key`/`*.pem` from ever entering an image layer — secrets reach the container as env vars, never copied in. (`.env` isn't in the source tree anyway, but the belt-and-suspenders matters for anyone who keeps one there.)

**Compose extension — the "clean solution" for container-specific config:** the `api` service uses `env_file: .env` for the *shared* values (DB credentials) and an `environment:` block to override only what differs inside a container — `DB_HOST=db` (the compose service name, not `localhost`), `OLLAMA_URL=http://host.docker.internal:11434` (Ollama stays native on the Windows host), `API_HOST=0.0.0.0` (bind past the container loopback), `LOG_DIR=/app/logs`. This is why one `.env` serves both host-run scripts (`ingest.py`, evals) and the container without edits — the container-only values live in compose, never in `.env`. `depends_on: { db: { condition: service_healthy } }` reuses the db healthcheck that already existed. The host publish port follows `${API_PORT:-8000}`.

**Empty-corpus startup (design decision made during implementation):** the roadmap demo starts the container on a *fresh clone* — i.e. `compose up` before `ingest.py` has ever run — but the Phase 1 startup called `verify_corpus_compatible`, which *raises* when no corpus exists. That would crash-loop the container before you could ingest. Resolved by splitting the concern: a missing corpus is the normal fresh-clone state (log a warning, serve `503` on `/query` until a corpus appears), whereas a *mismatched* corpus is still a hard startup failure. `/query` re-checks readiness on request, so ingesting into the running stack needs no API restart. New `check_corpus_ready()` in `main.py` wraps the distinction; the fail-fast mismatch guard is preserved.

**Backstop webhook (`rag/notify.py`):** the global exception handler now fires an async, fire-and-forget notification when `ERROR_WEBHOOK_URL` is set (unset = feature off, the default). One payload carries both `text` (Slack) and `content` (Discord) so a single URL covers either service with no provider switch. Hard rules honored: the message carries only the exception **class name**, `error_id`, and route — never the stack trace or exception message (which can leak connection details); full detail stays in the logs under the `error_id`. A notification failure is caught and logged, never propagated — it must not break the error response it accompanies. The task is held in a module-level set so the event loop can't GC it mid-flight. Unit-tested (delivery, disabled path, HTTP-error and connection-failure swallowing) with `httpx.MockTransport`, the same injection pattern as `OllamaEmbedder`.

**Log persistence:** `LOG_DIR` (unset locally → stdout only) adds a `FileHandler` at `LOG_DIR/api.log`. The compose `api` service sets `LOG_DIR=/app/logs`, bind-mounted to `./logs` on the host — a **bind mount** rather than a named volume because logs are low-write and staying browsable from the host is the point. Persists beyond `compose down`. `./logs` is gitignored.

**Verified end to end through the container:** image built clean; `docker compose up -d --build` brought up db + api; startup log shows it reached the host-native Ollama and matched the ingested corpus. `GET /health` → `{"status":"ok"}`; `POST /query` ("gut microbial diversity vs. arterial stiffness") returned the grounded answer citing `[13714201::0]` at score 0.907 — identical to the Phase 1 host-run result, confirming `host.docker.internal` routing works. Guardrail 1 empty-query still returns 422; `whoami` in the container is `app` (uid 1000); `./logs/api.log` written on the host.

**Deferred as planned:** nothing from this phase. The webhook's real-URL delivery was exercised only via `MockTransport` (no live Slack/Discord workspace) — the parsing and failure paths are covered; a live smoke test is a one-line `.env` change when a workspace exists.

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
