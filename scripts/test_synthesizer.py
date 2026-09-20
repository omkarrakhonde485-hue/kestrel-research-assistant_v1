import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.synthesizer import synthesizer_node


state = {
    "question": "What happens when a property value exceeds the ingestion limit?",

    "verifier_verdict": "conflicting_evidence",

    "verified_claims": [
        {
            "claim": (
                "When a property value exceeds the ingestion limit, "
                "the documentation conflicts on whether it is truncated "
                "or rejected."
            ),
            "verdict": "conflicting_evidence",
            "chunk_ids": [
                "spec-ingest-api:1",
                "onboarding-guide:2",
            ],
        }
    ],

    "retrieved_chunks": [
        {
            "chunk_id": "spec-ingest-api:1",
            "text": (
                "Oversized property values are truncated and the event "
                "is flagged with truncated=true."
            ),
            "metadata": {
                "title": "Event Ingestion API Specification",
                "category": "engineering",
            },
        },
        {
            "chunk_id": "onboarding-guide:2",
            "text": (
                "Events that exceed the property size limit are rejected "
                "rather than truncated."
            ),
            "metadata": {
                "title": "Customer Onboarding Guide",
                "category": "onboarding",
            },
        },
    ],
}


result = synthesizer_node(state)

print("\nSynthesizer result:")
print("-" * 60)

print("Answer:")
print(result["final_answer"])

print("\nCitations:")

for citation in result["citations"]:
    print(
        f"- {citation['chunk_id']}: "
        f"{citation['title']}"
    )