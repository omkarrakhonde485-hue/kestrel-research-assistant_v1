import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.agents.state import ResearchState
from src.utils.llm import get_llm


DEFAULT_ANSWER = (
    "The available documentation does not establish an answer."
)


synthesizer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Synthesizer Agent for a research assistant answering
questions about Kestrel Labs internal documentation.

Your job is to produce the final answer using ONLY the verified evidence.

IMPORTANT RULES:

1. Do not introduce facts that are not present in the verified claims.
2. Do not use outside knowledge.
3. Do not make unsupported assumptions.
4. Every important factual statement must have a citation.
5. Citations must use the exact chunk_id values provided in the verified
   evidence.
6. Citation titles must use the exact title values provided in the
   verified evidence.
7. Never invent or reconstruct document titles.
8. If evidence is insufficient, clearly say that the documentation does
   not establish the answer.
9. If evidence conflicts, clearly explain the disagreement and cite the
   sources on both sides.
10. Be concise and directly answer the user's question.
11. Do not mention internal agent names.
12. Do not mention that you are an AI agent.

Citation format inside the answer:

Growth projects can create up to 60 Beacons. [spec-beacons:4]

Return ONLY a JSON object.

The JSON object must contain exactly these two fields:

{{
  "answer": "The final answer with citations.",
  "citations": [
    {{
      "chunk_id": "exact chunk id",
      "title": "exact title from verified evidence"
    }}
  ]
}}

Do not use Markdown code fences.
Do not add any text before or after the JSON.

IMPORTANT:
Return valid JSON using double quotes.
Do not put unescaped newlines inside JSON string values.
""",
        ),
        (
            "human",
            """
User question:

{question}

Verifier verdict:

{verifier_verdict}

Verified evidence:

{verified_claims}
""",
        ),
    ]
)


def _extract_json(text: str) -> dict:
    """
    Extract a JSON object from the model response.

    Handles:
    - pure JSON
    - Markdown code fences
    - explanatory text before/after the JSON
    """

    if not text:
        raise ValueError("Synthesizer returned empty output.")

    text = text.strip()

    # Remove Markdown fences if the model added them.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.strip()

    # First attempt: complete response is JSON.
    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    # Second attempt: locate the outermost JSON object.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]

        try:
            parsed = json.loads(candidate)

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse synthesizer JSON: {exc}\n"
                f"Raw output:\n{text}"
            ) from exc

    raise ValueError(
        "Synthesizer did not return a JSON object.\n"
        f"Raw output:\n{text}"
    )


def _build_evidence_lookup(
    verified_claims: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Build authoritative chunk_id -> title mapping from verifier output.
    """

    evidence_lookup: dict[str, str] = {}

    for claim in verified_claims:
        if not isinstance(claim, dict):
            continue

        evidence_items = claim.get("evidence", [])

        if not isinstance(evidence_items, list):
            continue

        for evidence in evidence_items:
            if not isinstance(evidence, dict):
                continue

            chunk_id = evidence.get("chunk_id")
            title = evidence.get("title")

            if chunk_id:
                evidence_lookup[str(chunk_id)] = (
                    str(title) if title else ""
                )

    return evidence_lookup


def _normalise_citations(
    citations: Any,
    evidence_lookup: dict[str, str],
) -> list[dict[str, str]]:
    """
    Accept only citations whose chunk_id exists in verified evidence.

    Titles are always replaced with the authoritative verifier title.
    """

    if not isinstance(citations, list):
        return []

    valid_citations: list[dict[str, str]] = []
    seen: set[str] = set()

    for citation in citations:
        if not isinstance(citation, dict):
            continue

        chunk_id = citation.get("chunk_id")

        if not chunk_id:
            continue

        chunk_id = str(chunk_id)

        if chunk_id not in evidence_lookup:
            continue

        if chunk_id in seen:
            continue

        seen.add(chunk_id)

        valid_citations.append(
            {
                "chunk_id": chunk_id,
                "title": evidence_lookup[chunk_id],
            }
        )

    return valid_citations


def _citations_from_verified_claims(
    verified_claims: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Deterministically collect authoritative citations from verified evidence.

    Used when the LLM omits citations or returns malformed citation objects.
    """

    evidence_lookup = _build_evidence_lookup(
        verified_claims
    )

    return [
        {
            "chunk_id": chunk_id,
            "title": title,
        }
        for chunk_id, title in evidence_lookup.items()
    ]


def _claim_text(claim: dict[str, Any]) -> str:
    """
    Extract textual claim from possible verifier field names.
    """

    for key in ("claim", "statement", "text"):
        value = claim.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _build_fallback_answer(
    verifier_verdict: str,
    verified_claims: list[dict[str, Any]],
) -> str:
    """
    Build a deterministic grounded answer if LLM synthesis fails.

    This fallback uses only verifier-produced claims.
    """

    if verifier_verdict == "insufficient_evidence":
        return DEFAULT_ANSWER

    claim_lines: list[str] = []

    for claim in verified_claims:
        if not isinstance(claim, dict):
            continue

        claim_text = _claim_text(claim)

        if not claim_text:
            continue

        evidence_items = claim.get("evidence", [])

        if not isinstance(evidence_items, list):
            evidence_items = []

        chunk_ids: list[str] = []

        for evidence in evidence_items:
            if not isinstance(evidence, dict):
                continue

            chunk_id = evidence.get("chunk_id")

            if chunk_id:
                chunk_id = str(chunk_id)

                if chunk_id not in chunk_ids:
                    chunk_ids.append(chunk_id)

        if chunk_ids:
            citations = " ".join(
                f"[{chunk_id}]"
                for chunk_id in chunk_ids
            )

            claim_lines.append(
                f"{claim_text} {citations}"
            )
        else:
            claim_lines.append(claim_text)

    if not claim_lines:
        return DEFAULT_ANSWER

    if verifier_verdict == "conflicting_evidence":
        return (
            "The documentation contains conflicting statements:\n\n"
            + "\n".join(
                f"- {line}"
                for line in claim_lines
            )
        )

    if verifier_verdict == "partially_supported":
        return (
            "The available documentation only partially establishes "
            "the answer:\n\n"
            + "\n".join(
                f"- {line}"
                for line in claim_lines
            )
        )

    return "\n".join(claim_lines)


def _ensure_answer_has_citations(
    answer: str,
    citations: list[dict[str, str]],
) -> str:
    """
    Ensure the final answer contains references to verified chunk IDs.

    If the LLM omitted inline citations, append the missing verified IDs.
    """

    if not answer or not answer.strip():
        return DEFAULT_ANSWER

    answer = answer.strip()

    if not citations:
        return answer

    cited_ids = {
        match.group(1)
        for match in re.finditer(
            r"\[([^\]]+)\]",
            answer,
        )
    }

    missing_ids = [
        citation["chunk_id"]
        for citation in citations
        if citation["chunk_id"] not in cited_ids
    ]

    if not missing_ids:
        return answer

    suffix = " " + " ".join(
        f"[{chunk_id}]"
        for chunk_id in missing_ids
    )

    return answer + suffix


def _validate_answer_result(
    result: dict[str, Any],
) -> tuple[str, Any]:
    """
    Extract the expected fields from the model response.
    """

    answer = result.get(
        "answer",
        DEFAULT_ANSWER,
    )

    if not isinstance(answer, str):
        answer = str(answer)

    citations = result.get(
        "citations",
        [],
    )

    return answer.strip(), citations


def synthesizer_node(state: ResearchState) -> dict:
    print("\n[SYNTHESIZER INPUT]")

    verifier_verdict = state.get(
        "verifier_verdict",
        "insufficient_evidence",
    )

    verified_claims = state.get(
        "verified_claims",
        [],
    )

    print("Verifier verdict:")
    print(verifier_verdict)

    print("\nVerified evidence:")
    print(
        json.dumps(
            verified_claims,
            indent=2,
            default=str,
        )
    )

    evidence_lookup = _build_evidence_lookup(
        verified_claims
    )

    authoritative_citations = _citations_from_verified_claims(
        verified_claims
    )

    llm = get_llm()

    # IMPORTANT:
    #
    # Do NOT use:
    #
    # llm.bind(response_format={"type": "json_object"})
    #
    # Groq can reject this request with:
    #
    # json_validate_failed
    #
    # Instead, JSON is requested through the prompt and parsed locally.

    chain = synthesizer_prompt | llm

    payload = {
        "question": state["question"],
        "verifier_verdict": verifier_verdict,
        "verified_claims": json.dumps(
            verified_claims,
            indent=2,
            default=str,
        ),
    }

    result: dict[str, Any] | None = None

    try:
        response = chain.invoke(payload)

        raw_output = response.content

        if isinstance(raw_output, list):
            raw_output = "".join(
                str(item)
                for item in raw_output
            )

        raw_output = str(raw_output).strip()

        print("\n[SYNTHESIZER RAW OUTPUT]")
        print(raw_output)

        result = _extract_json(raw_output)

    except Exception as exc:
        print("\n[SYNTHESIZER WARNING]")
        print(
            "LLM synthesis failed or returned invalid JSON:"
        )
        print(exc)

        print("\n[SYNTHESIZER FALLBACK]")

    if result is None:
        answer = _build_fallback_answer(
            verifier_verdict,
            verified_claims,
        )

        unique_citations = authoritative_citations

    else:
        answer, model_citations = _validate_answer_result(
            result
        )

        unique_citations = _normalise_citations(
            model_citations,
            evidence_lookup,
        )

        # If the model omitted citations, use authoritative verifier
        # citations instead.
        if not unique_citations and authoritative_citations:
            unique_citations = authoritative_citations

        answer = _ensure_answer_has_citations(
            answer,
            unique_citations,
        )

    print("\n[SYNTHESIZER RESULT]")

    print("Answer:")
    print(answer)

    print("\nCitations:")

    for citation in unique_citations:
        print(
            f"- {citation['chunk_id']}: "
            f"{citation['title']}"
        )

    return {
        "final_answer": answer,
        "citations": unique_citations,
    }