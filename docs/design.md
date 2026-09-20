````markdown
# 🔎 Kestrel Research Assistant — Design Document

> **Retrieve first. Verify the evidence. Then answer.**

## 1. Overview

The Kestrel Research Assistant is a multi-agent research system designed to answer questions over a fixed internal Kestrel Labs documentation corpus.

The system is intentionally designed around a **retrieve → verify → synthesize** workflow rather than allowing the language model to answer directly from its general knowledge.

The assistant:

- retrieves relevant documentation before generating an answer,
- uses multiple specialized agents,
- maintains shared state across the workflow,
- supports multi-turn conversations,
- identifies conflicting evidence,
- refuses to invent unsupported information,
- cites the exact evidence chunks used,
- traces the complete workflow through LangSmith.

### Corpus

The supplied Kestrel corpus contains:

- **154 chunks**
- **25 documents**
- Categories:
  - Product
  - Release Notes
  - Pricing
  - Engineering
  - Incident
  - Policy
  - Onboarding

The corpus is treated as read-only application knowledge.

---

# 2. Design Goals

The system was designed around five primary goals.

### 2.1 Grounded answers

Answers should be based on retrieved Kestrel documentation rather than unsupported model knowledge.

### 2.2 Evidence verification

Retrieved evidence is passed through a dedicated verifier before synthesis.

### 2.3 Conflict awareness

If authoritative documents disagree, the assistant should expose the conflict rather than silently choosing one interpretation.

### 2.4 Honest uncertainty

When the corpus does not contain enough evidence, the system should return:

`insufficient_evidence`

rather than inventing an answer.

### 2.5 Observable multi-agent execution

Planner, retrieval, verification, and synthesis should all be traceable so that failures can be diagnosed from individual workflow stages.

---

# 3. System Architecture

```text
                         ┌──────────────────┐
                         │      USER        │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  PLANNER /       │
                         │     ROUTER       │
                         │                  │
                         │ Understands the  │
                         │ question and     │
                         │ creates research │
                         │ queries          │
                         └────────┬─────────┘
                                  │
                         research_queries
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ RESEARCHER / RETRIEVER   │
                    │                          │
                    │ Dense Retrieval          │
                    │        +                 │
                    │ BM25 Retrieval           │
                    │        ↓                 │
                    │ RRF Fusion               │
                    └────────────┬─────────────┘
                                 │
                         retrieved_chunks
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ VERIFIER / CRITIC         │
                    │                          │
                    │ Checks claims against    │
                    │ retrieved evidence       │
                    │                          │
                    │ supported                │
                    │ partially_supported      │
                    │ conflicting_evidence     │
                    │ insufficient_evidence    │
                    └────────────┬─────────────┘
                                 │
                          verified_claims
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ SYNTHESIZER              │
                    │                          │
                    │ Produces concise answer  │
                    │ with grounded citations  │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ ANSWER + CITATIONS        │
                    └──────────────────────────┘


                 ───────────────────────────────
                         SHARED STATE
                 ───────────────────────────────

     conversation_id
     turn
     question
     research_queries
     retrieved_chunks
     verified_claims
     verifier_verdict
     answer
     citations
     scores
````

The workflow is implemented using **LangGraph**, with each agent operating on shared state.

---

# 4. Agent Architecture

The system uses four specialized agents.

## 4.1 Planner / Router

### Responsibility

The Planner converts the user's question into one or more research queries.

It determines whether the question requires:

* a single retrieval query,
* multiple related queries,
* information from different documentation areas,
* follow-up context from previous conversation turns.

### Important behavior

The planner preserves important identifiers exactly.

For example:

```text
4.1.1
```

must remain:

```text
4.1.1
```

rather than being shortened or normalized into another version.

This is important for documentation questions involving release versions, limits, product names, IDs, or dates.

### Output

The planner adds:

```text
research_queries
```

to the shared graph state.

---

# 5. Researcher / Retriever

The Researcher executes the planner's queries against the local Kestrel corpus.

The retrieval layer uses a hybrid strategy:

```text
                    User Query
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
       Dense Retrieval         BM25 Retrieval
             │                     │
             └──────────┬──────────┘
                        ▼
                 RRF Fusion
                        │
                        ▼
                Ranked Evidence
```

## 5.1 Dense Retrieval

Dense retrieval uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

with Chroma as the local vector store.

This captures semantic similarity between the question and documentation.

---

## 5.2 BM25 Retrieval

BM25 provides lexical retrieval over the corpus.

This is particularly useful when a question contains:

* exact product names,
* version numbers,
* configuration names,
* error messages,
* IDs,
* technical terminology.

---

## 5.3 Reciprocal Rank Fusion

Dense and BM25 rankings are combined using **Reciprocal Rank Fusion (RRF)**.

The implementation uses:

```text
RRF_K = 60
```

The purpose is to combine semantic matching with exact lexical matching.

This was especially useful for the evaluation case involving the conflicting documentation around oversized event properties.

---

## 5.4 Evidence pruning

The retrieval layer can generate multiple candidate results across planner queries.

Before sending evidence to downstream agents:

* duplicate chunks are removed,
* evidence is aggregated across queries,
* matched queries are tracked,
* retrieval scores are preserved,
* the final evidence set is bounded.

The current downstream evidence cap is:

```text
MAX_RETRIEVED_CHUNKS = 8
```

This keeps the verifier and synthesizer context bounded while preserving useful retrieval recall.

---

# 6. Shared State and Agent Handoffs

LangGraph provides a shared state object through which agents communicate.

Conceptually:

```text
State
├── conversation_id
├── turn
├── question
├── research_queries
├── retrieved_chunks
├── verified_claims
├── verifier_verdict
├── answer
├── citations
└── scores
```

The handoff sequence is:

```text
Planner
   │
   │ research_queries
   ▼
Researcher
   │
   │ retrieved_chunks
   ▼
Verifier
   │
   │ verified_claims
   │ verifier_verdict
   ▼
Synthesizer
   │
   │ answer
   │ citations
   ▼
User
```

This separates responsibilities and makes intermediate decisions inspectable.

---

# 7. Verifier / Critic

The verifier is the main grounding and safety layer of the research workflow.

It receives:

* the original question,
* retrieved evidence,
* the candidate claims that can be supported by that evidence.

It checks whether each claim is actually supported.

## Verdict categories

The verifier uses four explicit verdicts:

### `supported`

The retrieved evidence directly supports the claim.

### `partially_supported`

Only part of the claim is supported by the available evidence.

### `conflicting_evidence`

Relevant retrieved documents provide contradictory information.

### `insufficient_evidence`

The retrieved corpus does not contain enough evidence to support the requested claim.

---

# 8. Conflict Handling

Conflict detection is a deliberate requirement of the system.

The supplied corpus contains a known contradiction concerning oversized event properties.

One document:

```text
spec-ingest-api:1
```

states that oversized property values are truncated and flagged with:

```text
truncated=true
```

Another document:

```text
onboarding-guide:2
```

states that oversized events are rejected.

The system is expected to surface this disagreement rather than selecting one statement without qualification.

During evaluation, the system successfully identified this case as:

```text
conflicting_evidence
```

and retained citations to the relevant evidence.

This demonstrates that the verifier operates independently from the final answer generation step.

---

# 9. Unsupported Questions

The system explicitly handles questions for which the supplied corpus does not provide sufficient evidence.

The intended flow is:

```text
Question
   ↓
Retrieve
   ↓
Evidence insufficient
   ↓
Verifier
   ↓
insufficient_evidence
   ↓
Synthesizer
   ↓
Transparent limitation
```

The assistant should not fill the gap using general model knowledge.

The evaluation includes a deliberately unsupported question, which was correctly handled with:

```text
insufficient_evidence
```

---

# 10. Synthesizer

The Synthesizer receives verified claims and produces the final response.

Its responsibilities are:

1. answer the user's question,
2. stay within the verified evidence,
3. preserve uncertainty or conflict,
4. attach citations,
5. avoid inventing unsupported facts.

Citations use the corpus chunk identifier format:

```text
[chunk-id]
```

For example:

```text
[spec-beacons:4]
```

The final response therefore provides a direct path from the generated answer back to the retrieved source chunk.

---

# 11. Multi-Turn Conversations

The system maintains conversation-level state including:

```text
conversation_id
turn
```

This allows follow-up questions to be interpreted in the context of an existing conversation.

The evaluation set includes follow-up questions to verify that the workflow can operate across multiple turns rather than treating every question as an isolated request.

---

# 12. LLM Layer

Generation uses Groq through LangChain.

Configured model:

```text
openai/gpt-oss-20b
```

Generation settings include:

```text
temperature = 0
max_tokens = 700
max_retries = 2
```

Temperature zero is used to reduce unnecessary variability in research responses.

The application does not depend on the model to perform retrieval itself. Retrieval and verification remain explicit workflow stages.

---

# 13. Reliability and Error Handling

The system includes bounded retries around model operations.

The design avoids unbounded retry loops.

The verifier allows a bounded number of attempts and falls back conservatively when a valid verification result cannot be produced.

The application also automatically creates the local Chroma index when the collection is missing.

This is important for deployment environments such as Streamlit Cloud, where generated local database files are not committed to Git.

The deployment flow is therefore:

```text
Application starts
       │
       ▼
Check Chroma collection
       │
   ┌───┴────┐
   │        │
exists    missing
   │        │
   │        ▼
   │   Build index
   │   from corpus
   │        │
   └───┬────┘
       ▼
   Run application
```

---

# 14. Observability

The entire workflow is traced using **LangSmith**.

The project records the major workflow stages, including:

```text
Planner
Researcher / Retrieval
Verifier
Synthesizer
```

This makes it possible to inspect:

* planner queries,
* retrieved evidence,
* verifier decisions,
* generated answers,
* execution failures,
* representative unsupported and multi-hop cases.

LangSmith traces are also recorded with the evaluation results where available.

Observability is treated as part of the system design rather than as a debugging-only feature.

---

# 15. Evaluation Design

The evaluation set contains:

```text
15 questions
```

covering multiple categories:

* Single-hop
* Multi-hop
* Conflict
* Unsupported
* Follow-up / multi-turn

Each evaluation question records:

```text
qid
question
type
conversation_id
turn
expected_answer
expected_chunk_ids
```

Evaluation outputs record:

```text
qid
answer
citations
retrieved_chunk_ids
verifier_verdict
scores
latency_seconds
langsmith_run_url
```

---

# 16. Evaluation Metrics

The evaluation measures:

### Retrieval Recall

Whether the expected evidence chunks were retrieved.

### Retrieval Precision

How much of the retrieved evidence corresponds to expected evidence.

### Citation Precision

Whether cited chunks are relevant to the generated answer.

### Citation Recall

Whether relevant expected evidence is represented in the citations.

### Answer Match

Whether the generated answer matches the expected answer sufficiently.

### Execution Error

Whether the workflow failed to complete successfully.

The final evaluation run completed all 15 questions without execution failures.

---

# 17. Evaluation Results

Final evaluation results:

| Metric              | Result |
| ------------------- | -----: |
| Retrieval Recall    |  0.929 |
| Retrieval Precision |  0.246 |
| Citation Precision  |  0.472 |
| Citation Recall     |  0.833 |
| Answer Match        |  0.689 |
| Execution Error     |  0.000 |

The results show that the retrieval system generally recovered the expected evidence, while precision and answer/citation quality remain areas for future improvement.

The evaluation also verified the system's behavior on the deliberately conflicting evidence case and the unsupported-question case.

---

# 18. Design Tradeoffs

## Hybrid retrieval vs dense-only retrieval

A dense-only retriever provides strong semantic matching but can be less reliable for exact technical identifiers and terminology.

BM25 was therefore added alongside dense retrieval.

The two rankings are fused using RRF.

### Tradeoff

This increases retrieval complexity and can introduce additional candidates, but improves the system's ability to recover exact technical evidence.

---

## Bounded evidence vs unrestricted context

The researcher keeps a bounded final evidence set.

### Tradeoff

Sending every potentially relevant chunk to the verifier could increase context size and noise.

Bounding the evidence at eight chunks reduces downstream context while retaining high retrieval recall in the evaluation.

---

## Dedicated verifier vs direct generation

A simpler architecture could retrieve documents and immediately ask the LLM to answer.

The current system deliberately adds a verifier.

### Tradeoff

This introduces another LLM call and therefore additional latency and cost.

The benefit is explicit support checking and conflict detection before synthesis.

---

## Local vector store vs hosted vector database

Chroma is used locally.

### Tradeoff

This simplifies deployment and keeps the corpus local, but does not provide the scalability or managed infrastructure of a hosted vector database.

For the fixed 154-chunk corpus, a local store is sufficient.

---

# 19. Security and Configuration

Secrets are not committed to the repository.

Environment configuration is provided through:

```text
.env
```

with an example configuration in:

```text
.env.example
```

The repository excludes:

```text
.env
.venv/
chroma_db/
__pycache__/
```

The corpus itself remains unchanged.

---

# 20. Deployment

The application is packaged with Docker and deployed through Streamlit Community Cloud.

Application entry point:

```text
app.py
```

The application listens on:

```text
8501
```

The Chroma index is automatically rebuilt from the committed corpus when required.

This allows the deployed application to operate without committing the generated vector database.

---

# 21. Project Structure

```text
kestrel-research-assistant/
│
├── corpus.jsonl
│
├── src/
│   ├── agents/
│   │   ├── state.py
│   │   ├── planner.py
│   │   ├── researcher.py
│   │   ├── verifier.py
│   │   └── synthesizer.py
│   │
│   ├── retrieval/
│   │   ├── dense.py
│   │   └── bootstrap.py
│   │
│   ├── utils/
│   │   └── llm.py
│   │
│   └── graph.py
│
├── scripts/
│   ├── validate_corpus.py
│   ├── build_index.py
│   ├── test_retrieval.py
│   ├── test_planner.py
│   ├── test_researcher.py
│   ├── test_verifier.py
│   ├── test_synthesizer.py
│   ├── test_graph.py
│   └── run_evaluation.py
│
├── results/
│   ├── eval_questions.jsonl
│   ├── eval_results.jsonl
│   ├── metrics_summary.json
│   └── improvement.md
│
├── docs/
│   └── design.md
│
├── app.py
├── main.py
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# 22. End-to-End Workflow

A typical request follows this sequence:

```text
1. User asks a question
          ↓
2. Planner creates research queries
          ↓
3. Researcher performs hybrid retrieval
          ↓
4. Evidence is deduplicated and bounded
          ↓
5. Verifier checks claims against evidence
          ↓
6. Conflicts / unsupported claims are classified
          ↓
7. Synthesizer produces the grounded response
          ↓
8. Citations are attached
          ↓
9. LangSmith records the workflow
          ↓
10. User receives answer + evidence references
```

The central design principle is:

> **The model does not get to answer first and justify later. Evidence is retrieved and verified before synthesis.**

---

# 23. Future Improvements

The evaluation results identify several possible improvements:

1. Improve retrieval precision through stronger candidate filtering.
2. Improve citation precision by restricting citations to claims directly supported by verified evidence.
3. Improve answer matching through more deterministic synthesis templates.
4. Add stronger query decomposition for complex multi-hop questions.
5. Add reranking after hybrid retrieval.
6. Expand the evaluation set beyond the initial 15 questions.
7. Add automated regression evaluation to CI.
8. Track latency separately for planning, retrieval, verification, and synthesis.

These improvements are intentionally separated from the current baseline so that future changes can be evaluated against the existing measurements.

---

# 24. Summary

The Kestrel Research Assistant implements a four-agent research workflow:

```text
Planner
   ↓
Researcher / Hybrid Retriever
   ↓
Verifier / Critic
   ↓
Synthesizer
```

The system combines:

* LangGraph shared state,
* local Chroma retrieval,
* Dense + BM25 + RRF hybrid search,
* evidence verification,
* explicit conflict detection,
* unsupported-question handling,
* citation-based grounding,
* multi-turn conversation state,
* LangSmith observability,
* bounded retries,
* Streamlit deployment.

The resulting architecture prioritizes **traceability, groundedness, and honest uncertainty** over simply generating fluent answers.

```

### What changed

- Created the complete **system design document**.
- Documented the **4-agent architecture** and LangGraph handoffs.
- Documented **Dense + BM25 + RRF** retrieval.
- Documented the deliberate **conflict case**.
- Documented **unsupported-question handling**.
- Documented **multi-turn state** and citations.
- Included the **actual final evaluation metrics**.
- Included reliability, deployment, observability, and design tradeoffs.
- Kept it aligned with the implementation we already completed.

**Next:** `results/improvement.md` — this is important because the assignment explicitly asks for **one concrete system change based on evaluation/traces with before/after results**.
```
