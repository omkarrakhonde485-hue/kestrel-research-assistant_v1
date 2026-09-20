from typing import Any, Dict

from src.agents.state import ResearchState
from src.retrieval.dense import get_dense_retriever


MAX_RETRIEVED_CHUNKS = 8


def researcher_node(state: ResearchState) -> dict:
    search_plan = state.get("search_plan", [])

    if not search_plan:
        return {
            "retrieved_chunks": []
        }

    retriever = get_dense_retriever(top_k=5)

    candidates: Dict[str, Dict[str, Any]] = {}

    for query in search_plan:
        results = retriever.search(query)

        for chunk in results:
            chunk_id = chunk["chunk_id"]

            if chunk_id not in candidates:
                candidates[chunk_id] = {
                    **chunk,
                    "matched_queries": [query],
                    "retrieval_score": chunk.get(
                        "rrf_score",
                        0.0,
                    ),
                }

            else:
                existing = candidates[chunk_id]

                # Keep track of every planner query that
                # independently retrieved this chunk.
                if query not in existing["matched_queries"]:
                    existing["matched_queries"].append(query)

                # Multiple retrieval queries finding the
                # same chunk is additional evidence that
                # the chunk is relevant.
                existing["retrieval_score"] = (
                    existing["retrieval_score"]
                    + chunk.get("rrf_score", 0.0)
                )

                # Preserve the strongest RRF score.
                if chunk.get("rrf_score", 0.0) > existing.get(
                    "rrf_score",
                    0.0,
                ):
                    existing["rrf_score"] = chunk[
                        "rrf_score"
                    ]

    # -----------------------------------------------------
    # Global ranking
    # -----------------------------------------------------
    #
    # Before this step, results were appended query-by-query.
    # That could give the verifier 15+ chunks for a 3-query
    # question.
    #
    # Now we rank all candidates globally and keep only the
    # strongest evidence.
    # -----------------------------------------------------

    ranked_chunks = sorted(
        candidates.values(),
        key=lambda chunk: (
            len(chunk["matched_queries"]),
            chunk.get("retrieval_score", 0.0),
        ),
        reverse=True,
    )

    ranked_chunks = ranked_chunks[
        :MAX_RETRIEVED_CHUNKS
    ]

    return {
        "retrieved_chunks": ranked_chunks
    }