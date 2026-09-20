import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


CORPUS_PATH = Path("corpus.jsonl")
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "kestrel_corpus"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_corpus():
    records = []

    with CORPUS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def main():
    print("Loading corpus...")

    records = load_corpus()

    print(f"Loaded {len(records)} chunks.")

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Creating ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Kestrel Labs research corpus"},
    )

    documents = []
    metadatas = []
    ids = []

    for record in records:
        ids.append(record["chunk_id"])
        documents.append(record["text"])

        metadatas.append(
            {
                "doc_id": record["doc_id"],
                "title": record["title"],
                "category": record["category"],
                "owner": record["owner"],
                "source_url": record["source_url"],
                "published": record["published"],
                "version": record["version"],
            }
        )

    print("Generating embeddings...")

    embeddings = model.encode(
        documents,
        show_progress_bar=True,
    ).tolist()

    print("Adding documents to ChromaDB...")

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print()
    print("Indexing complete.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Documents indexed: {collection.count()}")
    print(f"Database location: {CHROMA_PATH}")


if __name__ == "__main__":
    main()