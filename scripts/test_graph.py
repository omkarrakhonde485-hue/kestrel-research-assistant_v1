import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from src.graph import build_graph


app = build_graph()

initial_state = {
    "question": "How many Beacons can a Growth plan create?",
    "conversation_history": [],
}

print("=" * 60)
print("RUNNING GRAPH")
print("=" * 60)

result = app.invoke(initial_state)

print("\n" + "=" * 60)
print("FINAL STATE")
print("=" * 60)

print("\nQuestion:")
print(result.get("question"))

print("\nSearch plan:")
print(result.get("search_plan"))

print("\nRetrieved chunks:")
print(len(result.get("retrieved_chunks", [])))

print("\nVerified claims:")
print(result.get("verified_claims"))

print("\nVerifier verdict:")
print(result.get("verifier_verdict"))

print("\nFinal answer:")
print(result.get("final_answer"))

print("\nCitations:")
print(result.get("citations"))