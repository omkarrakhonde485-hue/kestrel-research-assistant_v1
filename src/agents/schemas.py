from typing import List, Literal

from pydantic import BaseModel, Field


Verdict = Literal[
    "supported",
    "partially_supported",
    "conflicting_evidence",
    "insufficient_evidence",
]


class VerifiedClaim(BaseModel):
    claim: str
    verdict: Verdict
    chunk_ids: List[str] = Field(default_factory=list)


class VerifierResult(BaseModel):
    verified_claims: List[VerifiedClaim] = Field(
        default_factory=list
    )
    overall_verdict: Verdict


class Citation(BaseModel):
    chunk_id: str
    title: str


class SynthesisResult(BaseModel):
    answer: str
    citations: List[Citation] = Field(
        default_factory=list
    )