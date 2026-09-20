import json
import re

from langchain_core.prompts import ChatPromptTemplate

from src.agents.state import ResearchState
from src.utils.llm import get_llm


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

The JSON object must contain these two fields:

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

    text = text.strip()

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
    )

    text = text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse synthesizer JSON: {exc}\n"
                f"Raw output:\n{text}"
            ) from exc

    raise ValueError(
        "Synthesizer did not return a JSON object.\n"
        f"Raw output:\n{text}"
    )


def synthesizer_node(state: ResearchState) -> dict:

    print("\n[SYNTHESIZER INPUT]")

    print("Verifier verdict:")
    print(
        state.get(
            "verifier_verdict",
            "insufficient_evidence",
        )
    )

    print("\nVerified evidence:")
    print(
        json.dumps(
            state.get(
                "verified_claims",
                [],
            ),
            indent=2,
            default=str,
        )
    )

    llm = get_llm()

    json_llm = llm.bind(
        response_format={
            "type": "json_object"
        }
    )

    chain = synthesizer_prompt | json_llm

    verified_claims = state.get(
        "verified_claims",
        [],
    )

    verifier_verdict = state.get(
        "verifier_verdict",
        "insufficient_evidence",
    )

    response = chain.invoke(
        {
            "question": state["question"],
            "verifier_verdict": verifier_verdict,
            "verified_claims": json.dumps(
                verified_claims,
                indent=2,
                default=str,
            ),
        }
    )

    raw_output = response.content.strip()

    print("\n[SYNTHESIZER RAW OUTPUT]")
    print(raw_output)

    result = _extract_json(
        raw_output
    )

    answer = result.get(
        "answer",
        "The available documentation does not establish an answer.",
    )

    citations = result.get(
        "citations",
        [],
    )

    valid_citations = []

    # Build authoritative citation lookup from verifier evidence.
    evidence_lookup = {}

    for claim in verified_claims:

        for evidence in claim.get(
            "evidence",
            [],
        ):

            chunk_id = evidence.get(
                "chunk_id"
            )

            title = evidence.get(
                "title"
            )

            if chunk_id:
                evidence_lookup[chunk_id] = title

    # Only accept citations that were actually verified.
    for citation in citations:

        if not isinstance(
            citation,
            dict,
        ):
            continue

        chunk_id = citation.get(
            "chunk_id"
        )

        if not chunk_id:
            continue

        if chunk_id not in evidence_lookup:
            continue

        valid_citations.append(
            {
                "chunk_id": chunk_id,
                "title": evidence_lookup[chunk_id],
            }
        )

    # Remove duplicate citations while preserving order.
    unique_citations = []

    seen_citations = set()

    for citation in valid_citations:

        key = citation["chunk_id"]

        if key in seen_citations:
            continue

        seen_citations.add(key)

        unique_citations.append(
            citation
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