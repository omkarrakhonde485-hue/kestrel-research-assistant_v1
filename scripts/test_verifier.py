import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.verifier import verifier_node


state = {
    "question": "What happens when a property value exceeds the ingestion limit?",
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

result = verifier_node(state)

print("\nVerifier result:")
print("-" * 60)

print("Overall verdict:")
print(result["verifier_verdict"])

print("\nVerified claims:")

for claim in result["verified_claims"]:
    print(f"\nClaim: {claim['claim']}")
    print(f"Verdict: {claim['verdict']}")
    print(f"Chunks: {claim['chunk_ids']}")