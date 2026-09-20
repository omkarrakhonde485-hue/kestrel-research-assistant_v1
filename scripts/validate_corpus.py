import json
from pathlib import Path


CORPUS_PATH = Path("corpus.jsonl")


def main():
    if not CORPUS_PATH.exists():
        raise FileNotFoundError("corpus.jsonl not found at project root.")

    records = []

    with CORPUS_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            records.append(record)

    print(f"Total chunks: {len(records)}")

    required_fields = {
        "chunk_id",
        "doc_id",
        "title",
        "category",
        "owner",
        "source_url",
        "published",
        "version",
        "text",
    }

    for index, record in enumerate(records, start=1):
        missing = required_fields - record.keys()

        if missing:
            raise ValueError(
                f"Record {index} is missing fields: {missing}"
            )

    chunk_ids = [record["chunk_id"] for record in records]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk_id found.")

    document_ids = {record["doc_id"] for record in records}
    categories = {record["category"] for record in records}

    print(f"Unique documents: {len(document_ids)}")
    print(f"Categories: {sorted(categories)}")
    print("Corpus validation: OK")


if __name__ == "__main__":
    main()