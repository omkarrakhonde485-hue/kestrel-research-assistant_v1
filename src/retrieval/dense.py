import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "kestrel_corpus"
CORPUS_PATH = "corpus.jsonl"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DENSE_CANDIDATES = 8
BM25_CANDIDATES = 8

RRF_K = 60

DEFAULT_TOP_K = 5


class HybridRetriever:
    """
    Hybrid local retriever combining:
    - Dense semantic retrieval from Chroma
    - BM25 lexical retrieval from the local corpus
    - Reciprocal Rank Fusion (RRF)

    The public interface remains:
        retriever.search(query)
    """

    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k

        self.model = SentenceTransformer(EMBEDDING_MODEL)

        self.client = chromadb.PersistentClient(
            path=CHROMA_PATH
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.chunk_ids: List[str] = []

        self._load_corpus()

        self.bm25 = BM25Okapi(
            [self._tokenize(text) for text in self.documents]
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        Lightweight BM25 tokenization.

        Lowercasing makes lexical retrieval case-insensitive.
        """
        return text.lower().split()

    @staticmethod
    def _build_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the corpus record's top-level metadata fields into
        the metadata shape used by the rest of the application.

        The corpus stores metadata fields at the top level rather
        than inside a nested "metadata" object.
        """

        metadata = record.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        # Preserve an existing nested metadata object if present,
        # while also copying the actual top-level corpus fields.
        for key in [
            "doc_id",
            "title",
            "category",
            "owner",
            "source_url",
            "published",
            "version",
        ]:
            if key in record:
                metadata[key] = record[key]

        return metadata

    def _load_corpus(self) -> None:
        """Load the original corpus without modifying it."""

        corpus_path = Path(CORPUS_PATH)

        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus file not found: {CORPUS_PATH}"
            )

        with corpus_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                self.chunk_ids.append(
                    record["chunk_id"]
                )

                self.documents.append(
                    record["text"]
                )

                self.metadata.append(
                    self._build_metadata(record)
                )

    def _dense_search(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve candidates using Chroma dense embeddings."""

        query_embedding = self.model.encode(
            [query]
        ).tolist()[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(
                DENSE_CANDIDATES,
                self.collection.count(),
            ),
        )

        chunks = []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(ids)):
            metadata = metadatas[i] or {}

            chunks.append(
                {
                    "chunk_id": ids[i],
                    "text": documents[i],
                    "metadata": metadata,
                    "distance": distances[i],
                    "retrieval_source": "dense",
                }
            )

        return chunks

    def _bm25_search(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """Retrieve candidates using local BM25."""

        tokens = self._tokenize(query)

        scores = self.bm25.get_scores(tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        chunks = []

        for index in ranked_indices[
            :BM25_CANDIDATES
        ]:
            chunks.append(
                {
                    "chunk_id": self.chunk_ids[index],
                    "text": self.documents[index],
                    "metadata": self.metadata[index],
                    "bm25_score": float(
                        scores[index]
                    ),
                    "retrieval_source": "bm25",
                }
            )

        return chunks

    @staticmethod
    def _rrf_score(rank: int) -> float:
        """
        Reciprocal Rank Fusion contribution.

        rank is 1-based.
        """
        return 1.0 / (RRF_K + rank)

    @staticmethod
    def _merge_metadata(
        existing: Dict[str, Any],
        incoming: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge metadata from different retrieval sources.

        Existing values are preserved unless they are missing or
        empty. This prevents BM25/Chroma metadata differences from
        causing citation metadata to disappear.
        """

        merged = dict(existing or {})

        for key, value in (incoming or {}).items():
            if (
                key not in merged
                or merged[key] in (None, "")
            ):
                merged[key] = value

        return merged

    def search(
        self,
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Run dense + BM25 retrieval and fuse results
        using Reciprocal Rank Fusion.
        """

        dense_results = self._dense_search(query)
        bm25_results = self._bm25_search(query)

        fused: Dict[str, Dict[str, Any]] = {}

        # -------------------------------------------------
        # Dense contribution
        # -------------------------------------------------

        for rank, chunk in enumerate(
            dense_results,
            start=1,
        ):
            chunk_id = chunk["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                    "dense_distance": None,
                    "distance": None,
                    "bm25_score": None,
                    "retrieval_sources": [],
                }

            fused[chunk_id]["rrf_score"] += (
                self._rrf_score(rank)
            )

            fused[chunk_id]["dense_rank"] = rank

            fused[chunk_id]["dense_distance"] = (
                chunk.get("distance")
            )

            # Preserve the original retriever interface.
            fused[chunk_id]["distance"] = (
                chunk.get("distance")
            )

            fused[chunk_id]["metadata"] = (
                self._merge_metadata(
                    fused[chunk_id]["metadata"],
                    chunk.get("metadata", {}),
                )
            )

            if (
                "dense"
                not in fused[chunk_id]["retrieval_sources"]
            ):
                fused[chunk_id][
                    "retrieval_sources"
                ].append("dense")

        # -------------------------------------------------
        # BM25 contribution
        # -------------------------------------------------

        for rank, chunk in enumerate(
            bm25_results,
            start=1,
        ):
            chunk_id = chunk["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                    "dense_distance": None,
                    "distance": None,
                    "bm25_score": None,
                    "retrieval_sources": [],
                }

            fused[chunk_id]["rrf_score"] += (
                self._rrf_score(rank)
            )

            fused[chunk_id]["bm25_rank"] = rank

            fused[chunk_id]["bm25_score"] = (
                chunk.get("bm25_score")
            )

            fused[chunk_id]["metadata"] = (
                self._merge_metadata(
                    fused[chunk_id]["metadata"],
                    chunk.get("metadata", {}),
                )
            )

            if (
                "bm25"
                not in fused[chunk_id]["retrieval_sources"]
            ):
                fused[chunk_id][
                    "retrieval_sources"
                ].append("bm25")

        # -------------------------------------------------
        # Final ranking
        # -------------------------------------------------

        ranked_chunks = sorted(
            fused.values(),
            key=lambda chunk: chunk["rrf_score"],
            reverse=True,
        )

        return ranked_chunks[: self.top_k]


# ---------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------

DenseRetriever = HybridRetriever


@lru_cache(maxsize=4)
def get_dense_retriever(
    top_k: int = DEFAULT_TOP_K,
):
    """
    Backward-compatible factory.

    Existing code calls get_dense_retriever(), so the
    researcher agent does not need to change.
    """

    return HybridRetriever(
        top_k=top_k
    )