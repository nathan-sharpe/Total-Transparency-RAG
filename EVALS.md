# EVALS.md

The living evaluation record for this RAG system. **Tier 1** (this file's current
content) measures *retrieval*: did the right documents come back? **Tier 2**
(added in Phase 3) measures *generation*: given the chunks, was the answer good?

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
