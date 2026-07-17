# EVALS.md

The living evaluation record for this RAG system. **Tier 1** measures
*retrieval*: did the right documents come back? **Tier 2** (added in Phase 3)
measures *generation*: given the chunks, was the answer good?

This is a changelog, not a report card. Every measured change gets a dated entry
in the format **change → hypothesis → before/after → conclusion**, added when the
change is made — never retro-written at the end. Numbers are only comparable
within the same config; the config snapshot is part of every result.

---

## Corpus

**SciFact** (BEIR format): 5,183 scientific-abstract documents. Chunked at
`CHUNK_SIZE=200` / `CHUNK_OVERLAP=40` words → **8,184 chunks** (most abstracts are
one or two chunks). Embedded with **nomic-embed-text** (768-dim) via Ollama.

## Golden set

SciFact's labeled **test** split: **300 queries**, loaded by `evals/golden.py` as
the join of `qrels/test.tsv` (which query, which relevant documents) with
`queries.jsonl` (the query text). Labels are stored as a **list** of relevant
`doc_id`s per query — singleton in most SciFact cases, but the list shape means
multi-relevant metrics need no schema change.

Alongside them, `evals/golden.py` carries a small hand-authored set of **8
out-of-domain queries** (cooking, sports, trivia) with no relevant document, used
only to measure the no-answer path.

## Credit rule (the methodological choice reviewers look for)

Labels are **document-level**; retrieval is **chunk-level**. So:

> A retrieved chunk counts as relevant **iff its `doc_id` is in the query's
> relevant-document list.**

Consequences, made explicit because they affect every number below:

- **`k` is counted in chunks, not documents.** `recall@5` looks at the top 5
  retrieved *chunks*. Two chunks from the same document collapse to one document
  when scoring recall (a document found twice is one document found).
- **MRR uses the rank of the first chunk whose document is relevant** (chunk
  position, 1-indexed).

The credit rule lives in the runner (`evals/run_retrieval.py`), which passes each
retrieved chunk's `doc_id` in rank order into the pure metric functions
(`evals/metrics.py`); the metrics themselves are generic over IDs and unit-tested
against hand-computed fixtures.

## Reproducing

```
docker compose up -d          # Postgres/pgvector
python ingest.py --dataset scifact   # if the corpus isn't already ingested
python evals/run_retrieval.py        # writes evals/results/retrieval.json + prints
```

The runner imports `rag/retrieval.py` directly (no HTTP), so it measures exactly
the code the API serves. Output JSON carries the full config snapshot and is the
artifact the Phase 5 CI gate will parse.

---

## Baseline — 2026-07-16

Ollama profile (`nomic-embed-text`, 768-dim), `CHUNK_SIZE=200`/`OVERLAP=40`,
`retrieve_k=10`, full 300-query test split. Source: `evals/results/retrieval.json`.

| Metric | Value |
|---|---|
| recall@5 | **0.5535** |
| recall@10 | **0.6024** |
| MRR | **0.4671** |

Reading these: retrieval surfaces a relevant document within the top 5 chunks for
~55% of queries and within the top 10 for ~60%; MRR ~0.47 means the first relevant
document typically lands around rank 2. The gap between recall@5 and recall@10 is
small (~5 points), so most of the gettable relevant documents are already in the
top 5 — pulling more chunks buys little. This is the number every later tuning
experiment (chunk size, top-k, embedding model — Phase 6) moves against.

---

## Changelog

### 2026-07-16 — Guardrail 2: no-answer path added

**Change.** The query flow now refuses *before* calling the generator when the
best retrieved chunk's cosine similarity is below `NO_ANSWER_THRESHOLD` (initial
value **0.60**), returning the canonical refusal string instead of an answer.
Mechanism: `is_answerable()` in `rag/retrieval.py`, wired into `main.py`.

**Hypothesis.** Out-of-domain queries retrieve only weakly-similar chunks, so a
similarity floor should refuse most of them while leaving genuine in-domain
queries — which retrieve strongly-similar chunks — untouched (no false refusals).

**Before / after** (measured on the 300 in-domain golden queries + 8 out-of-domain
queries; "before" = no guardrail, the generator is always called):

| | Before (no gate) | After (threshold 0.60) |
|---|---|---|
| Out-of-domain queries refused | 0 / 8 (0%) | **6 / 8 (75%)** |
| In-domain queries falsely refused | 0 / 300 (0%) | **0 / 300 (0%)** |

**Conclusion.** At 0.60 the gate refuses 75% of out-of-domain queries with **zero**
false refusals on in-domain queries — a clean win over always generating. Two
out-of-domain queries still slip through (their top chunk scored ≥ 0.60), and
because *every* in-domain query cleared 0.60, the threshold has headroom to rise.
Finding the point that catches those two without inducing in-domain false refusals
is the **Phase 6 threshold-sweep experiment** (refusal rate vs. false-refusal
rate); 0.60 is the conservative starting value, not a tuned one.

---

# Tier 2 — Generation quality (LLM-as-judge)

Tier 1 asks *did the right documents come back?* Tier 2 asks *given those
documents, was the answer good?* A hand-rolled judge (`evals/judge.py`, no
RAGAS) scores each answer on two rubric dimensions, 1–5:

- **faithfulness** — is every claim in the answer supported by the retrieved
  chunks? (judged only against the chunks, not the judge's own knowledge)
- **relevance** — does the answer actually address the question?

RAGAS's third headline metric, context *precision*, is deliberately **not**
judged: Tier 1 already measures retrieval against real golden-set labels, which
is stronger evidence than asking an 8B model to guess relevance.

**Judge independence.** The judge is `qwen2.5:7b` — a *different* model from the
generator (`llama3.1:8b`) — so the generator never scores its own output. The
6 GB VRAM budget fits one 7–8B model at a time, so `run_evals.py` runs in two
passes (generate all answers, then judge all answers) rather than swapping
models in and out of VRAM per query.

**8B-judge noise — read every number below with this caveat.** A local 7B judge
is a *weak* evaluator: it is noisy, and known to reward fluency and surface
agreement. These scores are directional signal for tracking the effect of
changes over time, **not** ground-truth answer quality. A one-time re-score with
a hosted frontier model (key via `.env` only) is the documented way to calibrate
them and is left as an optional Phase 3 stretch. Treat a 0.3-point movement in
mean faithfulness as noise, not a result.

## How the score is produced (methodology)

Per golden query: retrieve top-5 → **guardrail 2** (refuse below the 0.60
similarity threshold, generator never called) → generate → **guardrail 3b**
(strip citations pointing outside the retrieved set) → **judge**. Refusals — the
guardrail's *or* the generator's own — are **counted, not judged**: the judge
scores answers, and "was this refusal correct?" is a separate question the
refusal counts speak to. The judge's output is schema-validated with one retry
(**guardrail 3a**); a case that still fails is excluded and counted in
`judge_failures`, never averaged in.

## Baseline — 2026-07-17

Generator `llama3.1:8b` (temp 0), judge `qwen2.5:7b` (temp 0),
`nomic-embed-text` retrieval, `top_k=5`, `no_answer_threshold=0.60`, full
300-query test split. Source: `evals/results/generation.json`. Wall time: ~52 min
generation + ~21 min judging.

| | Count |
|---|---|
| Queries | 300 |
| Answered (judged) | **109** |
| Refused by generator | **191** |
| Refused by guardrail 2 | 0 |
| Judge schema-failures | 0 |

**Scores over the 109 answered queries:**

| Metric | Value |
|---|---|
| mean faithfulness | **3.93 / 5** |
| mean relevance | **4.33 / 5** |
| faithfulness ≥ 4 rate | **67.0%** |
| answers with ungrounded citations | **0** |

Score distributions (judged answers): faithfulness `{1:4, 3:32, 4:37, 5:36}`
(no 2s); relevance `{1:4, 2:1, 3:22, 4:10, 5:72}`.

**Reading these — the two tiers together tell the real story.** The dominant
result is that **the generator refuses 191/300 (64%) queries even though
guardrail 2 refused none** — every query cleared the 0.60 retrieval threshold
(the generator-refused set had top similarity 0.65–0.88, i.e. relevant chunks
*were* retrieved), yet Llama 3.1 8B declined to answer. The cause is structural:
SciFact "queries" are **declarative claims** ("0-dimensional biomaterials show
inductive properties"), not questions, and the generator — under a prompt that
tells it to refuse when the context is insufficient — treats a claim it can't
cleanly confirm as unanswerable. This is exactly what Tier 2 exists to expose:
Tier 1 says retrieval surfaces a relevant document ~55% of the time, but the
end-to-end system only *answers* 36% of the time, and that gap is a generation
problem, not a retrieval one.

For the answers it *does* produce, faithfulness 3.93 and relevance 4.33 (72 of
109 relevance scores are a perfect 5) say the generator is conservative but
mostly grounded when it commits — consistent with a model erring toward refusal.
**Zero ungrounded citations** across all 109 answers means guardrail 3b had
nothing to strip: the generator never cited a chunk ID outside the retrieved set
(encouraging, though the guardrail's value is being there for the case that does
occur). **Zero judge schema-failures** across 109 calls means the constrained
decoding + validation held without a single retry-to-failure.

Every Phase 6 generation experiment (prompt variants, top-k, chunk size) moves
against these numbers. The obvious first lever the baseline points at: the
refusal rate is a **prompt** artifact (claims-as-questions), and a generator
prompt variant that reframes the claim as "assess this statement against the
context" is the highest-leverage measured experiment to try — tracked to Phase 6
so it's run as a before/after, not a casual edit.

---

## Tier 2 changelog

### 2026-07-17 — Guardrail 3a: judge output schema-validated

**Change.** All judge output is constrained to the `JudgeVerdict` JSON schema at
the Ollama API level (`format=`) *and* Pydantic-validated on receipt. On a
validation failure the judge retries **once** with the validation error fed back
into the conversation; a second failure raises `JudgeError`, and the runner
excludes that case and counts it in `judge_failures`. Schema bounds
(`1 ≤ score ≤ 5`) are enforced by validation, not just the schema shape.

**Hypothesis.** A local 7B model will occasionally emit malformed or
out-of-range verdicts; without enforcement those silently corrupt the averages.
Constrained decoding plus validation-with-retry should drive usable-verdict rate
to ~100% while guaranteeing no garbage is ever averaged in.

**Before / after.** No "before" — the guardrail ships with the judge. Measured
on the 109 judged answers: **0 judge failures**, 0 retries-to-failure. Constrained
decoding did the heavy lifting; the retry path is the backstop for when it
doesn't.

**Conclusion.** Verdicts enter the metrics only if they are well-formed and
in-range, by construction. The mechanism is proven present (schema enforced,
`judge_failures` reported loudly) even though this clean corpus didn't exercise
the failure path — which is the point of a guardrail.

### 2026-07-17 — Guardrail 3b: citation grounding at serve time

**Change.** After generation, `ground_citations()` verifies every cited chunk ID
against the retrieved set. IDs pointing outside it (hallucinated citations) are
**stripped from the answer text** and returned separately; the API surfaces them
in a new `ungrounded_citations` field so a non-empty list flags the response as
suspect. The prompt *asks* the model to cite only provided chunks; this enforces
it in code. (Full per-claim faithfulness checking at serve time is the expensive
strong version — documented, not built; that is what the Tier-2 judge approximates
offline.)

**Hypothesis.** The generator will sometimes cite chunk IDs it wasn't given;
stripping them prevents a fabricated `[doc::idx]` from lending false authority to
an answer.

**Before / after.** Measured on the 109 answered baseline queries: **0 answers
with ungrounded citations, 0 IDs stripped.** The generator cited only real
retrieved IDs throughout.

**Conclusion.** On this corpus the generator's citation discipline is already
clean, so the guardrail is currently a no-op in aggregate — but it is the code
that catches the failure when the prompt doesn't, and it is unit-tested against
hallucinated-ID fixtures (`tests/test_generation.py`) rather than relying on the
corpus to exercise it.
