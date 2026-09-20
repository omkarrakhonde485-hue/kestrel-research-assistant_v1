````markdown
# 🔧 System Improvement — Hybrid Retrieval

## 1. Problem Identified

The initial retrieval system used dense vector retrieval only.

While semantic retrieval worked well for general questions, technical documentation questions can depend heavily on exact terms such as:

- version numbers,
- API names,
- configuration names,
- technical identifiers,
- exact error terminology.

A dense-only approach can sometimes rank semantically similar chunks above a chunk containing the exact terminology required by the question.

This was particularly important for the Kestrel corpus because the evaluation contains technical questions and a deliberate contradiction involving oversized event properties.

---

## 2. Change Made

The retrieval layer was changed from:

```text
Dense Retrieval
      ↓
Top-k Results
````

to:

```text
                 Query
                   │
          ┌────────┴────────┐
          ▼                 ▼
   Dense Retrieval      BM25 Retrieval
          │                 │
          └────────┬────────┘
                   ▼
             RRF Fusion
                   │
                   ▼
            Ranked Results
```

The new retriever combines:

* Dense semantic retrieval using `all-MiniLM-L6-v2`
* BM25 lexical retrieval
* Reciprocal Rank Fusion (RRF)

The RRF constant used by the implementation is:

```text
RRF_K = 60
```

The final evidence passed to downstream agents is bounded to reduce unnecessary context.

---

## 3. Why This Change Was Made

The goal was to improve retrieval robustness for technical documentation.

Dense retrieval is useful for understanding semantic similarity.

BM25 complements it by rewarding exact lexical matches.

Combining both allows the system to benefit from:

```text
Semantic similarity
        +
Exact terminology
        ↓
More robust retrieval
```

This was especially useful for questions where the evidence depends on exact technical wording.

---

## 4. Evidence From the Conflict Case

The corpus intentionally contains conflicting documentation about oversized event properties.

The relevant chunks are:

```text
spec-ingest-api:1
onboarding-guide:2
```

The question:

> When an event property exceeds the 4 KB limit, is it truncated or rejected?

requires the system to retrieve evidence from both documents.

After the hybrid retrieval change, the retrieval system recovered both relevant chunks.

The downstream verifier then classified the evidence as:

```text
conflicting_evidence
```

This allowed the final system to surface the disagreement rather than silently selecting one source.

---

## 5. Before vs After

### Before

```text
Retrieval strategy:
Dense-only

Problem:
Exact technical terminology could be harder to retrieve reliably.

Conflict test:
The two contradictory evidence chunks were not reliably recovered together.
```

### After

```text
Retrieval strategy:
Dense + BM25 + RRF

Improvement:
Semantic and lexical retrieval signals are combined.

Conflict test:
Both spec-ingest-api:1 and onboarding-guide:2 were recovered.

Verifier result:
conflicting_evidence
```

---

## 6. Evaluation Results

The final evaluation was run on:

```text
15 questions
```

covering:

* single-hop questions,
* multi-hop questions,
* conflicting evidence,
* unsupported questions,
* follow-up questions.

Final aggregate results:

| Metric              | Final Result |
| ------------------- | -----------: |
| Retrieval Recall    |        0.929 |
| Retrieval Precision |        0.246 |
| Citation Precision  |        0.472 |
| Citation Recall     |        0.833 |
| Answer Match        |        0.689 |
| Execution Error     |        0.000 |

The evaluation demonstrates that the final system successfully completed all 15 evaluation cases without execution failures.

---

## 7. Why This Improvement Was Kept

The hybrid approach adds some retrieval complexity compared with dense-only retrieval, but it provides complementary retrieval signals.

For this fixed technical documentation corpus, the additional BM25 stage is inexpensive and runs locally.

The improvement was therefore retained as part of the final architecture.

---

## 8. Future Improvement

The final evaluation still shows room to improve retrieval precision.

A future iteration could add a dedicated reranking stage after hybrid retrieval:

```text
Dense + BM25
      ↓
RRF
      ↓
Reranker
      ↓
Final Evidence
```

This could reduce irrelevant retrieved chunks while preserving the recall benefits of hybrid retrieval.

This was not added to the final submission because the current implementation already satisfied the required workflow and evaluation was completed against the final system.

```