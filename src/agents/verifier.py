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

MAX_CHUNK_TEXT_CHARS = 2500
MAX_VERIFIER_ATTEMPTS = 2


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
- Always return at least one complete block.
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


retry_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict evidence verifier.

Use ONLY the retrieved evidence.

Return exactly ONE complete block in this format:

CLAIM: <answer-relevant factual claim>
VERDICT: <supported | partially_supported | conflicting_evidence | insufficient_evidence>
CHUNKS: <one or more exact chunk IDs>

Rules:
- If two retrieved sources disagree, use conflicting_evidence.
- If the evidence does not establish the answer, use insufficient_evidence.
- Never invent facts or chunk IDs.
- Return ONLY the three lines.
""",
        ),
        (
            "human",
            """
Question:
{question}

Evidence:
{retrieved_evidence}
""",
        ),
    ]
)


def _format_retrieved_evidence(chunks: list) -> str:
    formatted = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})

        chunk_id = chunk.get(
            "chunk_id",
            "unknown",
        )

        title = metadata.get(
            "title",
            "Unknown document",
        )

        published = metadata.get(
            "published",
            "unknown",
        )

        version = metadata.get(
            "version",
            "unknown",
        )

        text = chunk.get(
            "text",
            "",
        )

        if len(text) > MAX_CHUNK_TEXT_CHARS:
            text = text[:MAX_CHUNK_TEXT_CHARS].rstrip()
            text += "\n[truncated for verifier context]"

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


def _normalise_verdict(value: str) -> str:
    verdict = (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    if verdict not in ALLOWED_VERDICTS:
        return "insufficient_evidence"

    return verdict


def _valid_chunk_ids(
    raw_chunk_ids: list,
    retrieved_chunks: list,
) -> list:
    valid_ids = {
        chunk.get("chunk_id")
        for chunk in retrieved_chunks
        if chunk.get("chunk_id")
    }

    result = []

    for chunk_id in raw_chunk_ids:
        chunk_id = chunk_id.strip()

        if chunk_id and chunk_id in valid_ids:
            result.append(chunk_id)

    return result


def _parse_complete_blocks(
    text: str,
    retrieved_chunks: list,
) -> list:
    text = text.strip()

    text = re.sub(
        r"```(?:text|markdown)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace(
        "```",
        "",
    ).strip()

    text = text.replace(
        "\r\n",
        "\n",
    )

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

    matches = list(
        pattern.finditer(text)
    )

    print("\n[PARSER DEBUG]")
    print(
        f"Raw output length: {len(text)}"
    )
    print(
        f"Matched blocks: {len(matches)}"
    )

    claims = []

    for match in matches:
        claim = match.group(
            "claim"
        ).strip()

        verdict = _normalise_verdict(
            match.group("verdict")
        )

        raw_chunks = match.group(
            "chunks"
        ).strip()

        raw_chunk_ids = [
            chunk.strip()
            for chunk in raw_chunks.split(",")
            if chunk.strip()
        ]

        chunk_ids = _valid_chunk_ids(
            raw_chunk_ids,
            retrieved_chunks,
        )

        if not claim:
            continue

        # A supported / partial / conflict claim must point to
        # actual retrieved evidence.
        if (
            verdict != "insufficient_evidence"
            and not chunk_ids
        ):
            print(
                "[PARSER WARNING] "
                "Claim had no valid retrieved chunk IDs."
            )
            continue

        claims.append(
            {
                "claim": claim,
                "verdict": verdict,
                "chunk_ids": chunk_ids,
            }
        )

    return claims


def _extract_fallback_claim(
    text: str,
) -> str:
    match = re.search(
        r"CLAIM:\s*(.+?)(?:\n|$)",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return ""

    return match.group(1).strip()


def _infer_explicit_verdict(
    text: str,
) -> str:
    patterns = [
        (
            "conflicting_evidence",
            r"\bconflicting[_ -]?evidence\b",
        ),
        (
            "partially_supported",
            r"\bpartially[_ -]?supported\b",
        ),
        (
            "insufficient_evidence",
            r"\binsufficient[_ -]?evidence\b",
        ),
        (
            "supported",
            r"\bsupported\b",
        ),
    ]

    for verdict, pattern in patterns:
        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            return verdict

    return ""


def _parse_verifier_output(
    raw_output: str,
    retrieved_chunks: list,
) -> list:
    """
    Parse verifier output conservatively.

    Strict parsing is attempted first. A fallback can recover a
    truncated CLAIM line only when the model explicitly supplied
    an allowed verdict and valid retrieved chunk IDs.
    """

    text = raw_output.strip()

    if not text:
        return []

    claims = _parse_complete_blocks(
        text,
        retrieved_chunks,
    )

    if claims:
        print(
            "[PARSER MODE] strict"
        )
        return claims

    # Conservative fallback for malformed/truncated output.
    claim = _extract_fallback_claim(
        text
    )

    verdict = _infer_explicit_verdict(
        text
    )

    chunk_match = re.search(
        r"CHUNKS:\s*(.+?)(?:\n|$)",
        text,
        flags=re.IGNORECASE,
    )

    raw_chunk_ids = []

    if chunk_match:
        raw_chunk_ids = [
            chunk.strip()
            for chunk in chunk_match.group(1).split(",")
            if chunk.strip()
        ]

    chunk_ids = _valid_chunk_ids(
        raw_chunk_ids,
        retrieved_chunks,
    )

    if not verdict:
        print(
            "[PARSER MODE] fallback rejected: "
            "no explicit verdict"
        )
        return []

    if not claim:
        print(
            "[PARSER MODE] fallback rejected: "
            "no claim"
        )
        return []

    if (
        verdict != "insufficient_evidence"
        and not chunk_ids
    ):
        print(
            "[PARSER MODE] fallback rejected: "
            "no valid chunk IDs"
        )
        return []

    print(
        "[PARSER MODE] fallback"
    )

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
        chunk_id = chunk.get(
            "chunk_id"
        )

        if not chunk_id:
            continue

        metadata = chunk.get(
            "metadata",
            {},
        )

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


def _invoke_verifier(
    chain,
    question: str,
    retrieved_evidence: str,
    attempt: int,
):
    if attempt == 1:
        response = chain.invoke(
            {
                "question": question,
                "retrieved_evidence": retrieved_evidence,
            }
        )
    else:
        response = (
            retry_prompt
            | get_llm()
        ).invoke(
            {
                "question": question,
                "retrieved_evidence": retrieved_evidence,
            }
        )

    content = getattr(
        response,
        "content",
        "",
    )

    if isinstance(content, list):
        content = "".join(
            str(item)
            for item in content
        )

    return str(content).strip()


def verifier_node(
    state: ResearchState,
) -> dict:

    retrieved_chunks = state.get(
        "retrieved_chunks",
        [],
    )

    retrieved_evidence = (
        _format_retrieved_evidence(
            retrieved_chunks
        )
    )

    question = state["question"]

    verified_claims = []

    for attempt in range(
        1,
        MAX_VERIFIER_ATTEMPTS + 1,
    ):
        print(
            f"\n[VERIFIER ATTEMPT {attempt}/"
            f"{MAX_VERIFIER_ATTEMPTS}]"
        )

        llm = get_llm()

        chain = (
            verifier_prompt
            | llm
        )

        try:
            raw_output = _invoke_verifier(
                chain,
                question,
                retrieved_evidence,
                attempt,
            )
        except Exception as exc:
            print(
                "[VERIFIER ERROR]"
                f" attempt={attempt}: {exc}"
            )

            # Do not loop forever. If the first call fails,
            # make exactly one retry.
            if attempt < MAX_VERIFIER_ATTEMPTS:
                continue

            raw_output = ""

        print(
            "\n[VERIFIER RAW OUTPUT]"
        )
        print(raw_output)

        verified_claims = (
            _parse_verifier_output(
                raw_output,
                retrieved_chunks,
            )
        )

        if verified_claims:
            break

        if attempt < MAX_VERIFIER_ATTEMPTS:
            print(
                "[VERIFIER RETRY] "
                "No usable verifier claim was parsed."
            )

    verified_claims = (
        _attach_evidence_metadata(
            verified_claims,
            retrieved_chunks,
        )
    )

    overall_verdict = (
        _calculate_overall_verdict(
            verified_claims
        )
    )

    print(
        "\n[VERIFIED CLAIMS]"
    )

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
        f"[OVERALL VERDICT] "
        f"{overall_verdict}"
    )

    return {
        "verified_claims": verified_claims,
        "verifier_verdict": overall_verdict,
    }