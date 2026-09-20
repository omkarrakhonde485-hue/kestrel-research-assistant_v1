from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class ResearchState(TypedDict, total=False):
    # User input
    question: str

    # Previous conversation turns
    conversation_history: List[Dict[str, str]]

    # Planner output
    search_plan: List[str]

    # Retriever output
    retrieved_chunks: List[Dict[str, Any]]

    # Verifier output
    verified_claims: List[Dict[str, Any]]
    verifier_verdict: str

    # Final response
    final_answer: str
    citations: List[Dict[str, str]]