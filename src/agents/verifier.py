import re

from langchain_core.prompts import ChatPromptTemplate

from src.agents.state import ResearchState
from src.utils.llm import get_llm


ALLOWED_VERDICTS = {
    "supported",
    "partially_supported",
    "conflicting_evidence",
    "insufficient_evidence",
}


verifier_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Verifier Agent for a research assistant answering questions
about Kestrel Labs internal documentation.

Your job is to determine whether the retrieved evidence supports the
claims needed to answer the user's question.

Rules:

1. Use ONLY the provided retrieved documents.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Every material claim must have supporting or contradicting chunk IDs.
5. If two documents disagree, mark the claim as conflicting_evidence.
6. Do not silently reconcile conflicting documents.
7. If the evidence does not establish an answer, use insufficient_evidence.
8. Consider document dates and versions when evaluating conflicts.
9. Be conservative.
10. Only report claims relevant to the user's question.
11. Preserve exact numbers, dates, versions, limits, and identifiers from
    the retrieved evidence.
12. Keep claims concise so the complete output fits within the response limit.

Allowed verdicts:

supported
partially_supported
conflicting_evidence
insufficient_evidence

OUTPUT FORMAT:

CLAIM: <factual claim>
VERDICT: <one allowed verdict>
CHUNKS: <comma-separated chunk IDs>

Example:

CLAIM: A Growth plan can create 60 Beacons.
VERDICT: supported
CHUNKS: spec-beacons:4, pricing-plans:1

IMPORTANT:

- Return ONLY CLAIM / VERDICT / CHUNKS blocks.
- Do not use Markdown.
- Do not add explanations.
- Do not add JSON.
- Do not add numbering.
""",
        ),
        (
            "human",
            """
User question:

{question}

Retrieved evidence:

{retrieved_evidence}
""",
        ),
    ]
)


def _format_retrieved_evidence(chunks: list) -> str:
    formatted = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})

        chunk_id = chunk.get("chunk_id", "unknown")
        title = metadata.get("title", "Unknown document")
        published = metadata.get("published", "unknown")
        version = metadata.get("version", "unknown")
        text = chunk.get("text", "")

        formatted.append(
            "\n".join(
                [
                    f"CHUNK ID: {chunk_id}",
                    f"TITLE: {title}",
                    f"PUBLISHED: {published}",
                    f"VERSION: {version}",
                    "TEXT:",
                    text,
                ]
            )
        )

    if not formatted:
        return "No retrieved evidence was found."

    return "\n\n---\n\n".join(formatted)


def _normalise_verdict(value: str) -> str | None:
    verdict = (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    if verdict in ALLOWED_VERDICTS:
        return verdict

    return None


def _valid_chunk_ids(
    chunk_ids: list,
    retrieved_chunks: list,
) -> list:
    valid_ids = {
        chunk.get("chunk_id")
        for chunk in retrieved_chunks
        if chunk.get("chunk_id")
    }

    result = []

    for chunk_id in chunk_ids:
        chunk_id = chunk_id.strip()

        if chunk_id in valid_ids and chunk_id not in result:
            result.append(chunk_id)

    return result


def _parse_complete_blocks(
    text: str,
    retrieved_chunks: list,
) -> list:
    pattern = re.compile(
        r"""
        CLAIM:\s*(?P<claim>.*?)
        \n
        VERDICT:\s*(?P<verdict>[A-Za-z_ -]+)
        \n
        CHUNKS:\s*(?P<chunks>.*?)
        (?=
            \n\s*CLAIM:
            |
            \Z
        )
        """,
        re.IGNORECASE | re.VERBOSE | re.DOTALL,
    )

    matches = list(pattern.finditer(text))

    claims = []

    for match in matches:
        claim = match.group("claim").strip()

        verdict = _normalise_verdict(
            match.group("verdict")
        )

        raw_chunks = match.group("chunks").strip()

        chunk_ids = _valid_chunk_ids(
            raw_chunks.split(","),
            retrieved_chunks,
        )

        if not claim:
            continue

        if verdict is None:
            continue

        claims.append(
            {
                "claim": claim,
                "verdict": verdict,
                "chunk_ids": chunk_ids,
            }
        )

    return claims


def _extract_chunk_ids_from_text(
    text: str,
    retrieved_chunks: list,
) -> list:
    """
    Extract only chunk IDs that actually exist in retrieved evidence.

    This prevents the fallback parser from inventing citations.
    """

    available_ids = [
        chunk.get("chunk_id")
        for chunk in retrieved_chunks
        if chunk.get("chunk_id")
    ]

    found = []

    for chunk_id in available_ids:
        if re.search(
            rf"(?<![\w:-]){re.escape(chunk_id)}(?![\w:-])",
            text,
        ):
            found.append(chunk_id)

    return found


def _infer_verdict_from_text(
    text: str,
    retrieved_chunks: list,
) -> str | None:
    """
    Infer a verdict only when the model explicitly mentions one.

    This does not infer supported/conflicting status from arbitrary prose.
    """

    verdict_patterns = [
        (
            "conflicting_evidence",
            r"\bconflicting(?:_|\s+)evidence\b",
        ),
        (
            "partially_supported",
            r"\bpartially(?:_|\s+)supported\b",
        ),
        (
            "insufficient_evidence",
            r"\binsufficient(?:_|\s+)evidence\b",
        ),
        (
            "supported",
            r"\bsupported\b",
        ),
    ]

    lowered = text.lower()

    for verdict, pattern in verdict_patterns:
        if re.search(pattern, lowered):
            return verdict

    return None


def _extract_fallback_claim(
    text: str,
) -> str:
    """
    Recover a truncated CLAIM line when the normal block parser fails.
    """

    match = re.search(
        r"CLAIM:\s*(.+?)(?:\n|$)",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return ""

    return match.group(1).strip()


def _parse_verifier_output(
    raw_output: str,
    retrieved_chunks: list,
) -> list:
    """
    Parse verifier output using strict blocks first.

    If the model output is truncated or malformed, use a conservative
    fallback that can recover a claim only when the output itself contains
    a claim and references only retrieved chunk IDs.
    """

    text = raw_output.strip()

    text = re.sub(
        r"```(?:text|markdown)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace("```", "").strip()
    text = text.replace("\r\n", "\n")

    claims = _parse_complete_blocks(
        text,
        retrieved_chunks,
    )

    if claims:
        print("\n[PARSER DEBUG]")
        print(f"Raw output length: {len(text)}")
        print(f"Matched blocks: {len(claims)}")
        print("Parser mode: strict")

        return claims

    print("\n[PARSER DEBUG]")
    print(f"Raw output length: {len(text)}")
    print("Matched blocks: 0")
    print("Parser mode: fallback")

    if not text:
        return []

    claim = _extract_fallback_claim(text)

    if not claim:
        return []

    verdict = _infer_verdict_from_text(
        text,
        retrieved_chunks,
    )

    chunk_ids = _extract_chunk_ids_from_text(
        text,
        retrieved_chunks,
    )

    # A fallback claim without an explicit verdict or grounded chunks
    # is not safe to pass downstream.
    if verdict is None:
        return []

    if not chunk_ids and verdict != "insufficient_evidence":
        return []

    return [
        {
            "claim": claim,
            "verdict": verdict,
            "chunk_ids": chunk_ids,
        }
    ]


def _calculate_overall_verdict(
    verified_claims: list,
) -> str:
    verdicts = [
        claim.get("verdict")
        for claim in verified_claims
    ]

    if "conflicting_evidence" in verdicts:
        return "conflicting_evidence"

    if "partially_supported" in verdicts:
        return "partially_supported"

    if "supported" in verdicts:
        return "supported"

    return "insufficient_evidence"


def _attach_evidence_metadata(
    verified_claims: list,
    retrieved_chunks: list,
) -> list:
    """
    Attach authoritative document metadata to every verified chunk.

    The verifier identifies chunk IDs.
    The corpus metadata supplies the exact title.
    """

    chunk_lookup = {}

    for chunk in retrieved_chunks:
        chunk_id = chunk.get("chunk_id")

        if not chunk_id:
            continue

        metadata = chunk.get("metadata", {})

        chunk_lookup[chunk_id] = {
            "chunk_id": chunk_id,
            "title": metadata.get(
                "title",
                "Unknown document",
            ),
        }

    enriched_claims = []

    for claim in verified_claims:
        evidence = []

        for chunk_id in claim.get(
            "chunk_ids",
            [],
        ):
            metadata = chunk_lookup.get(
                chunk_id,
                {
                    "chunk_id": chunk_id,
                    "title": "Unknown document",
                },
            )

            evidence.append(metadata)

        enriched_claims.append(
            {
                "claim": claim["claim"],
                "verdict": claim["verdict"],
                "chunk_ids": claim["chunk_ids"],
                "evidence": evidence,
            }
        )

    return enriched_claims


def verifier_node(state: ResearchState) -> dict:
    llm = get_llm()

    chain = verifier_prompt | llm

    retrieved_chunks = state.get(
        "retrieved_chunks",
        [],
    )

    retrieved_evidence = _format_retrieved_evidence(
        retrieved_chunks
    )

    response = chain.invoke(
        {
            "question": state["question"],
            "retrieved_evidence": retrieved_evidence,
        }
    )

    raw_output = response.content.strip()

    print("\n[VERIFIER RAW OUTPUT]")
    print(raw_output)

    verified_claims = _parse_verifier_output(
        raw_output,
        retrieved_chunks,
    )

    verified_claims = _attach_evidence_metadata(
        verified_claims,
        retrieved_chunks,
    )

    overall_verdict = _calculate_overall_verdict(
        verified_claims
    )

    print("\n[VERIFIED CLAIMS]")

    for claim in verified_claims:
        print(
            f"Claim: {claim['claim']}"
        )

        print(
            f"Verdict: {claim['verdict']}"
        )

        print(
            f"Chunks: {claim['chunk_ids']}"
        )

        print(
            f"Evidence: {claim['evidence']}"
        )

        print()

    print(
        f"[OVERALL VERDICT] {overall_verdict}"
    )

    return {
        "verified_claims": verified_claims,
        "verifier_verdict": overall_verdict,
    }