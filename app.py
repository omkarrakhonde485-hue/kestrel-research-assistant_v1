import streamlit as st

from src.graph import build_graph


st.set_page_config(
    page_title="Kestrel Research Assistant",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 1.5rem 1.7rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 18px;
            margin-bottom: 1.5rem;
            background: linear-gradient(
                135deg,
                rgba(99, 102, 241, 0.10),
                rgba(59, 130, 246, 0.05)
            );
        }

        .hero h1 {
            margin-bottom: 0.35rem;
        }

        .hero p {
            margin-bottom: 0;
            color: #777;
        }

        .citation-card {
            padding: 0.85rem 1rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 12px;
            margin-bottom: 0.7rem;
        }

        .citation-id {
            font-family: monospace;
            font-weight: 600;
        }

        .verdict {
            padding: 0.8rem 1rem;
            border-radius: 12px;
            margin: 0.8rem 0 1.2rem 0;
            border: 1px solid rgba(128, 128, 128, 0.25);
        }

        .small-muted {
            color: #777;
            font-size: 0.85rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Graph
# ---------------------------------------------------------

@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔎 Kestrel")

    st.caption("Multi-Agent Research Assistant")

    st.divider()

    st.markdown("### Agent pipeline")

    st.markdown(
        """
        **1. Planner**  
        Decomposes the question

        **↓**

        **2. Researcher**  
        Searches the local corpus

        **↓**

        **3. Verifier**  
        Checks claims against evidence

        **↓**

        **4. Synthesizer**  
        Produces the cited answer
        """
    )

    st.divider()

    st.markdown("### Grounding")

    st.caption(
        "Answers are generated only from retrieved Kestrel "
        "documentation. Evidence is exposed as chunk IDs "
        "and document titles."
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.last_result = None
        st.rerun()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🔎 Kestrel Research Assistant</h1>
        <p>
            Ask questions about Kestrel Labs documentation and inspect
            the evidence behind every answer.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Conversation history
# ---------------------------------------------------------

for message in st.session_state.conversation_history:
    role = message.get("role", "assistant")
    content = message.get("content", "")

    with st.chat_message(role):
        st.markdown(content)


# ---------------------------------------------------------
# Question input
# ---------------------------------------------------------

question = st.chat_input(
    "Ask something about Kestrel Labs..."
)


if question:
    # Display user message immediately.
    with st.chat_message("user"):
        st.markdown(question)

    # Build conversation history for the graph.
    history = list(st.session_state.conversation_history)

    history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # -----------------------------------------------------
    # Agent progress
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        progress = st.status(
            "Running research workflow...",
            expanded=True,
        )

        try:
            progress.write("🧭 Planner → decomposing the question")
            progress.write("🔎 Researcher → retrieving evidence")
            progress.write("🛡️ Verifier → checking claims")
            progress.write("✍️ Synthesizer → preparing final answer")

            result = graph.invoke(
                {
                    "question": question,
                    "conversation_history": history,
                }
            )

            progress.update(
                label="Research workflow completed",
                state="complete",
                expanded=False,
            )

        except Exception as exc:
            progress.update(
                label="Workflow failed",
                state="error",
                expanded=True,
            )

            st.error(
                "The research workflow failed. "
                "Check the terminal logs for details."
            )

            st.exception(exc)

            st.stop()

        # -------------------------------------------------
        # Result
        # -------------------------------------------------

        answer = result.get(
            "final_answer",
            "The system did not return a final answer.",
        )

        citations = result.get(
            "citations",
            [],
        )

        verdict = result.get(
            "verifier_verdict",
            "insufficient_evidence",
        )

        verified_claims = result.get(
            "verified_claims",
            [],
        )

        st.markdown(answer)

        # -------------------------------------------------
        # Verdict
        # -------------------------------------------------

        verdict_labels = {
            "supported": "✅ Supported",
            "partially_supported": "🟡 Partially supported",
            "conflicting_evidence": "⚠️ Conflicting evidence",
            "insufficient_evidence": "❔ Insufficient evidence",
        }

        verdict_label = verdict_labels.get(
            verdict,
            verdict,
        )

        st.markdown(
            f"""
            <div class="verdict">
                <strong>Verifier status:</strong>
                {verdict_label}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # Citations
        # -------------------------------------------------

        if citations:
            st.markdown("### Sources")

            for citation in citations:
                chunk_id = citation.get(
                    "chunk_id",
                    "unknown",
                )

                title = citation.get(
                    "title",
                    "Unknown document",
                )

                st.markdown(
                    f"""
                    <div class="citation-card">
                        <div class="citation-id">
                            {chunk_id}
                        </div>
                        <div>
                            {title}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:
            st.caption(
                "No supporting citations were produced."
            )

        # -------------------------------------------------
        # Verifier details
        # -------------------------------------------------

        if verified_claims:
            with st.expander("Verifier details"):
                for index, claim in enumerate(
                    verified_claims,
                    start=1,
                ):
                    st.markdown(
                        f"**Claim {index}**"
                    )

                    st.write(
                        claim.get(
                            "claim",
                            "",
                        )
                    )

                    st.write(
                        f"**Verdict:** "
                        f"{claim.get('verdict', 'unknown')}"
                    )

                    chunk_ids = claim.get(
                        "chunk_ids",
                        [],
                    )

                    if chunk_ids:
                        st.write(
                            "**Evidence:** "
                            + ", ".join(chunk_ids)
                        )

        # -------------------------------------------------
        # Store conversation
        # -------------------------------------------------

        st.session_state.conversation_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        st.session_state.conversation_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        st.session_state.last_result = result