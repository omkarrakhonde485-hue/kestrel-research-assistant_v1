import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.dense import DenseRetriever


def main():
    retriever = DenseRetriever(top_k=5)

    question = "How does the Beacon scheduler handle missed evaluations?"

    results = retriever.search(question)

    print("\nQuery:")
    print(question)

    print("\nRetrieved chunks:\n")

    for rank, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank: {rank}")
        print(f"Chunk: {result['chunk_id']}")
        print(f"Title: {result['metadata']['title']}")
        print(f"Category: {result['metadata']['category']}")
        print(f"Distance: {result['distance']}")
        print(f"\n{result['text'][:500]}")


if __name__ == "__main__":
    main()