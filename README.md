````markdown
# 🔎 Kestrel Research Assistant

> **Retrieve first. Verify the evidence. Then answer.**

A multi-agent research assistant for Kestrel Labs' internal documentation, built with LangGraph, hybrid retrieval, evidence verification, grounded synthesis, and source-level citations.

---

## 🧠 Architecture

```text
                         👤 USER
                           │
                           ▼
                  ┌─────────────────┐
                  │ 🧠 Planner      │
                  │    / Router     │
                  │                 │
                  │ Decomposes the  │
                  │ question into   │
                  │ retrieval goals │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ 🔍 Researcher   │
                  │    / Retriever  │
                  │                 │
                  │ Dense + BM25    │
                  │      + RRF      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ 🛡️ Verifier     │
                  │    / Critic     │
                  │                 │
                  │ Checks claims   │
                  │ against evidence│
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ ✨ Synthesizer  │
                  │                 │
                  │ Produces a      │
                  │ grounded answer │
                  │ with citations  │
                  └────────┬────────┘
                           │
                           ▼
                   💬 ANSWER + SOURCES
````

The workflow is orchestrated using **LangGraph**, with the agents communicating through a shared research state.

---

## 🤖 Multi-Agent System

### 🧠 1. Planner / Router

Converts the user's question into a small set of high-value retrieval queries.

**Responsibilities:**

* Decompose complex questions
* Identify multiple evidence requirements
* Preserve important numbers, IDs, dates, versions, and technical terminology
* Handle multi-hop and conflicting-evidence questions

The planner does not answer the question.

### 🔍 2. Researcher / Retriever

Searches the Kestrel corpus using hybrid retrieval.

The retrieval pipeline combines:

* Dense semantic search using `sentence-transformers/all-MiniLM-L6-v2`
* BM25 lexical retrieval
* Reciprocal Rank Fusion (RRF)

This combination helps with both semantic similarity and exact technical terms, IDs, versions, and numeric constraints.

### 🛡️ 3. Verifier / Critic

Checks whether claims are actually supported by the retrieved documentation.

Possible verdicts:

```text
supported
partially_supported
conflicting_evidence
insufficient_evidence
```

The verifier also validates cited chunk IDs against retrieved evidence.

When the documentation does not provide enough evidence, the system is designed to say so instead of inventing an answer.

### ✨ 4. Synthesizer

Produces the final response using only verified evidence.

The synthesizer:

* Generates a concise grounded answer
* Attaches source citations
* Uses only verified chunk IDs
* Falls back to deterministic evidence-based synthesis when generation fails

---

## 📚 Grounded Answering

Every Kestrel-specific answer follows:

```text
❓ Question
     ↓
🔍 Retrieve evidence
     ↓
🛡️ Verify claims
     ↓
✨ Synthesize answer
     ↓
📌 Attach citations
```

Example:

```text
Question:
How many Beacons can a Growth plan create?

Answer:
A Growth plan can create 60 Beacons.

Sources:
spec-beacons:4
pricing-plans:1
rn-3-5:2
```

This makes it possible to trace an answer back to the exact corpus chunks supporting it.

---

## ⚔️ Conflict Handling

The corpus intentionally contains contradictory documentation.

One example concerns oversized event properties:

* `spec-ingest-api:1` describes oversized property values as being truncated and flagged.
* `onboarding-guide:2` describes oversized events as being rejected.

The verifier is designed to identify this as:

```text
conflicting_evidence
```

rather than silently selecting one statement.

---

## 🔄 Multi-Turn Conversations

The system maintains conversation history in the shared LangGraph state.

Example:

```text
👤 User:
How many Beacons can a Growth plan create?

🤖 Assistant:
A Growth plan can create 60 Beacons.

👤 User:
What about Scale?

🤖 Assistant:
...
```

The follow-up can use previous conversation context when constructing the retrieval plan.

---

## 🔍 Retrieval

### 🧬 Dense Retrieval

The system uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

for local embeddings.

Hosted embedding APIs are not used.

### 🔤 BM25

BM25 provides lexical retrieval over the original corpus and is particularly useful for:

* Version numbers
* Product names
* Chunk IDs
* Technical terminology
* Exact phrases

### 🔗 Reciprocal Rank Fusion

Dense and BM25 results are combined using Reciprocal Rank Fusion.

```text
🧬 Dense Retrieval
        +
     🔤 BM25
        │
        ▼
       🔗 RRF
        │
        ▼
📑 Hybrid Ranked Evidence
```

---

## 🗂️ Corpus

The provided Kestrel corpus contains:

* **154 chunks**
* **25 documents**

Categories include:

```text
product
release-notes
pricing
engineering
incident
policy
onboarding
```

`corpus.jsonl` is kept unchanged and is committed at the repository root.

---

## 📊 Evaluation

The system was evaluated against **15 questions** covering:

* 🎯 Single-hop questions
* 🔗 Multi-hop questions
* ⚔️ Conflicting evidence
* ❓ Unsupported questions
* 🔄 Follow-up questions

Evaluation outputs are stored in:

```text
results/
├── eval_questions.jsonl
├── eval_results.jsonl
├── metrics_summary.json
└── improvement.md
```

### 📈 Metrics

| Metric                 |    Result |
| ---------------------- | --------: |
| 🎯 Retrieval Recall    | **0.929** |
| 🔎 Retrieval Precision |     0.246 |
| 📌 Citation Precision  |     0.472 |
| 📚 Citation Recall     |     0.833 |
| 💬 Answer Match        |     0.689 |
| ❌ Execution Error      | **0.000** |

The evaluation informed retrieval and grounding improvements, including the move from dense-only retrieval to hybrid Dense + BM25 + RRF retrieval and improvements to evidence verification and synthesis.

Run the evaluation with:

```bash
python scripts/run_evaluation.py
```

---

## 🔭 Observability

The workflow is instrumented with **LangSmith**.

```text
🧠 Planner
    ↓
🔍 Researcher / Retrieval
    ↓
🛡️ Verifier
    ↓
✨ Synthesizer
```

LangSmith is used to inspect:

* Agent inputs and outputs
* Retrieval behaviour
* Verification decisions
* Latency
* Generation behaviour
* Representative workflow runs

Configure:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=kestrel-research-assistant
```

---

## 🗃️ Project Structure

```text
kestrel-research-assistant/
│
├── 📄 corpus.jsonl
├── 📄 app.py
├── 📄 main.py
├── 📄 requirements.txt
├── 📄 Dockerfile
├── 📄 .env.example
├── 📄 .gitignore
├── 📄 README.md
│
├── 🤖 src/
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
├── 🧪 scripts/
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
├── 📊 results/
│   ├── eval_questions.jsonl
│   ├── eval_results.jsonl
│   ├── metrics_summary.json
│   └── improvement.md
│
└── 📚 docs/
```

---

## 🛠️ Tech Stack

| Component            | Technology                               |
| -------------------- | ---------------------------------------- |
| 🐍 Language          | Python 3.11+                             |
| 🧩 Orchestration     | LangGraph                                |
| 🤖 Agent Framework   | LangChain                                |
| ⚡ LLM Provider       | Groq                                     |
| 🧠 Generation Model  | `openai/gpt-oss-20b`                     |
| 🗄️ Vector Store     | ChromaDB                                 |
| 🧬 Dense Embeddings  | `sentence-transformers/all-MiniLM-L6-v2` |
| 🔤 Lexical Retrieval | BM25                                     |
| 🔗 Ranking           | Reciprocal Rank Fusion                   |
| 🔭 Observability     | LangSmith                                |
| 🎨 Frontend          | Streamlit                                |
| ⚙️ Configuration     | python-dotenv                            |

---

## 🚀 Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/omkarrakhonde485-hue/kestrel-research-assistant_v1.git
cd kestrel-research-assistant_v1
```

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and configure:

```env
GROQ_API_KEY=your_groq_api_key

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=kestrel-research-assistant
```

> 🔐 **Never commit `.env` to Git.**

---

## 🗄️ Build the Local Index

Create the local Chroma index with:

```bash
python scripts/build_index.py
```

The application also contains a deployment bootstrap that creates the Chroma collection automatically when a fresh environment does not already contain it.

This allows deployment without committing the generated `chroma_db/` directory.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🧪 Run Tests

```bash
python scripts/test_retrieval.py
python scripts/test_planner.py
python scripts/test_researcher.py
python scripts/test_verifier.py
python scripts/test_synthesizer.py
python scripts/test_graph.py
```

---

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud** from the GitHub repository.

```text
🐙 GitHub
    │
    ▼
☁️ Streamlit Community Cloud
    │
    ▼
🎨 Streamlit
    │
    ▼
🧩 LangGraph Workflow
    │
    ▼
🤖 Multi-Agent Research System
```

Secrets such as `GROQ_API_KEY` and `LANGSMITH_API_KEY` are configured through the deployment platform rather than committed to the repository.

---

## ⚡ Reliability & Free-Tier Considerations

* 🧬 Local embeddings avoid hosted embedding costs
* 🔍 Retrieval is performed locally
* 🔄 Generation calls are sequential
* 📏 Output lengths are bounded
* 🔁 Groq retries are bounded
* 🛡️ Deterministic fallbacks are available for some generation failures
* 🗄️ Generated Chroma data is not committed to Git

---

## ⚖️ Design Trade-offs

### Why four agents?

The four-agent design separates:

```text
🧠 Planning
    ↓
🔍 Retrieval
    ↓
🛡️ Verification
    ↓
✨ Synthesis
```

This makes failures easier to inspect than a single LLM call.

### Why hybrid retrieval?

Dense retrieval handles semantic similarity, while BM25 is useful for exact identifiers, versions, numbers, and technical terminology.

### Why a verifier?

Retrieval can return relevant text without guaranteeing that the final claim is actually supported. The verifier creates an explicit checkpoint before answer generation.

### Why local embeddings?

The assignment requires local embeddings and this avoids additional hosted embedding API dependencies.

### Why Chroma?

Chroma provides a lightweight local vector store suitable for the 154-chunk corpus.

---

## 🔮 Future Improvements

* 🎯 Better retrieval precision through adaptive retrieval depth
* 📌 More precise citation selection
* 🔗 Improved handling of long multi-hop evidence sets
* 📊 More extensive evaluation coverage
* 🔭 Automated LangSmith dataset evaluation integration
* ⚡ Streaming agent progress to the UI
* 📖 More detailed source inspection in the frontend
* 💾 Persistent conversation storage

---

## 🔐 Security

Secrets are loaded from environment variables.

The following are intentionally excluded from Git:

```text
.env
.venv/
chroma_db/
__pycache__/
```

No API keys should be committed to the repository.

---

## 📜 License

This project was created as a take-home assignment implementation for the Kestrel Labs research assistant task.

````