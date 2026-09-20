import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.researcher import researcher_node


state = {
    "question": "How many Beacons can a Growth plan create?",
    "search_plan": [
        "Growth plan Beacon limit",
        "Beacon plan limits Starter Growth Scale",
    ],
}

result = researcher_node(state)

print("\nRetrieved chunks:")
print("-" * 60)

for i, chunk in enumerate(result["retrieved_chunks"], start=1):
    print(f"\nRank: {i}")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Title: {chunk['metadata'].get('title')}")
    print(f"Distance: {chunk['distance']:.4f}")
    print(f"Text: {chunk['text'][:500]}...")