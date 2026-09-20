from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "kestrel_corpus"


def ensure_chroma_index() -> None:
    """
    Ensure the local Chroma collection exists.

    On a fresh deployment (for example Streamlit Cloud), the generated
    chroma_db directory is not committed to Git. In that case, build the
    index from the read-only corpus.jsonl before the first research query.
    """

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))

        try:
            client.get_collection(name=COLLECTION_NAME)

            print(
                f"[INDEX] Chroma collection '{COLLECTION_NAME}' already exists."
            )
            return

        except Exception:
            print(
                f"[INDEX] Chroma collection '{COLLECTION_NAME}' not found."
            )

    except Exception as exc:
        print(f"[INDEX] Could not inspect Chroma collection: {exc}")

    build_script = PROJECT_ROOT / "scripts" / "build_index.py"

    if not build_script.exists():
        raise FileNotFoundError(
            f"Index build script not found: {build_script}"
        )

    corpus_path = PROJECT_ROOT / "corpus.jsonl"

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_path}"
        )

    print("[INDEX] Building Chroma index from corpus.jsonl...")
    print(f"[INDEX] Corpus: {corpus_path}")
    print(f"[INDEX] Database: {CHROMA_PATH}")

    env = os.environ.copy()

    result = subprocess.run(
        [sys.executable, str(build_script)],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)

        raise RuntimeError(
            "Chroma index build failed. "
            f"build_index.py exited with code {result.returncode}."
        )

    # Verify the collection really exists after the build.
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "Chroma index build completed, but the "
            f"'{COLLECTION_NAME}' collection was not created."
        ) from exc

    print(
        f"[INDEX] Chroma collection ready: "
        f"'{COLLECTION_NAME}' ({collection.count()} chunks)"
    )