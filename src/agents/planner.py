import re

from langchain_core.prompts import ChatPromptTemplate

from src.agents.state import ResearchState
from src.utils.llm import get_llm


planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Planner/Router Agent for a research assistant over
Kestrel Labs internal documentation.

Your job is to convert the user's question into a small set of
high-value retrieval queries.

You are NOT answering the question.

Your queries will be sent to a document retriever.

RULES:

1. Generate 2 to 3 retrieval queries.
2. For simple single-hop questions, prefer 2 queries.
3. For multi-hop questions, use 3 queries.
4. For conflict/version questions, use 3 queries.
5. Each query must target a DISTINCT piece of evidence.
6. Do NOT generate paraphrases of the same query.
7. Preserve EXACTLY:
   - numbers
   - dates
   - version numbers
   - IDs
   - product names
   - plan names
   - technical terms
8. Never shorten, rewrite, normalize, or partially replace a version number.
9. If the question contains a version such as 4.1.1, every query
   referring to that version must contain exactly 4.1.1.
10. If the question contains an ID, preserve that ID exactly.
11. If the question contains a date, preserve the date exactly.
12. If the question contains a numeric limit, preserve that number exactly.
13. If the question asks about an incident, separate queries may target:
    - what happened
    - root cause
    - resolution/fix
14. If the question asks about a product or limit, target:
    - the specification
    - the relevant pricing/plan documentation
    - release notes if version history matters
15. If the question asks about conflicting information, create queries
    that retrieve the sources likely to contain each side of the conflict.
16. Do not answer the user's question.
17. Return ONLY the retrieval queries, one per line.
18. Do not number the queries.
19. Do not use bullets.
20. Do not add explanations.

Example:

User question:
How many Beacons can a Growth plan create?

Good output:

Growth plan Beacon creation limit
Growth plan Beacon quota pricing

Bad output:

Growth plan Beacon creation limit
Growth plan number of Beacons allowed
Growth plan Beacon quota maximum
Growth plan Beacon creation maximum

The bad example contains four paraphrases of the same retrieval intent.

Another example:

User question:
What happened when the Beacon scheduler missed evaluations?

Good output:

Beacon scheduler missed evaluations incident
Beacon scheduler missed evaluations root cause
Beacon scheduler missed evaluations fix

Another example:

User question:
What happens when a property value exceeds the ingestion limit?

Good output:

ingestion property size limit behavior
oversized property values truncation
oversized property values rejection onboarding

Preserve exact technical terminology from the question whenever possible.
""",
        ),
        (
            "human",
            """
User question:

{question}

Conversation history:

{conversation_history}
""",
        ),
    ]
)


def _extract_protected_tokens(question: str) -> list[str]:
    """
    Extract exact tokens from the user's question that should not be
    accidentally rewritten by the planner.

    This focuses on versions, dates, IDs, and numeric values.
    """

    protected: list[str] = []

    patterns = [
        # Semantic versions: 4.1.1, 3.5, 4.0.3, etc.
        r"\b\d+\.\d+(?:\.\d+)+(?:[-+][A-Za-z0-9.-]+)?\b",

        # Decimal versions / numbers such as 4.1.
        r"\b\d+\.\d+\b",

        # Dates such as February 2026, 2026-02-15, etc.
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+\d{4}\b",

        # IDs containing letters/numbers/hyphens.
        r"\b[A-Za-z]+-\d{3,}[A-Za-z0-9-]*\b",

        # Numeric values.
        r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",
    ]

    for pattern in patterns:
        for match in re.findall(
            pattern,
            question,
            flags=re.IGNORECASE,
        ):
            token = str(match).strip()

            if token and token not in protected:
                protected.append(token)

    # Sort longest first so "4.1.1" is protected before shorter
    # fragments such as "4.1".
    protected.sort(
        key=len,
        reverse=True,
    )

    return protected


def _preserve_exact_tokens(
    query: str,
    question: str,
) -> str:
    """
    Correct planner output when it mutates an exact numeric/version token
    from the user's question.

    The method is intentionally conservative:
    it only replaces suspicious numeric/version tokens when the query
    appears to refer to the same token context.
    """

    protected_tokens = _extract_protected_tokens(
        question
    )

    if not protected_tokens:
        return query

    corrected = query

    # Version/date/ID tokens are handled first.
    for token in protected_tokens:
        if not re.search(
            r"\d",
            token,
        ):
            continue

        # A query that contains another numeric/version token near the
        # same textual context may have had the exact token rewritten.
        query_has_number = bool(
            re.search(
                r"\d",
                corrected,
            )
        )

        if not query_has_number:
            continue

        # Only correct when the query has a clear semantic relationship
        # to the protected token. For version numbers this is usually
        # indicated by words such as version, fix, release, incident,
        # scheduler, etc.
        token_is_version = bool(
            re.fullmatch(
                r"\d+\.\d+(?:\.\d+)+",
                token,
            )
        )

        if token_is_version:
            # If another dotted numeric token exists, replace it.
            dotted_numbers = re.findall(
                r"\b\d+\.\d+(?:\.\d+)+\b",
                corrected,
            )

            if dotted_numbers:
                for candidate in dotted_numbers:
                    if candidate != token:
                        corrected = corrected.replace(
                            candidate,
                            token,
                        )

            # Handle a model mutation such as 4.1.1 -> 1.1.
            # If the exact version isn't present but the query clearly
            # references a version/fix/release, append the exact version
            # rather than risk searching for the wrong release.
            if token not in corrected:
                version_context = bool(
                    re.search(
                        r"\b(version|release|fix|patch|upgrade|"
                        r"change|changed|scheduler)\b",
                        corrected,
                        flags=re.IGNORECASE,
                    )
                )

                if version_context:
                    corrected = (
                        f"{corrected} version {token}"
                    )

    return corrected.strip()


def planner_node(state: ResearchState) -> dict:
    llm = get_llm()

    chain = planner_prompt | llm

    conversation_history = state.get(
        "conversation_history",
        [],
    )

    response = chain.invoke(
        {
            "question": state["question"],
            "conversation_history": conversation_history,
        }
    )

    raw_output = response.content

    if isinstance(raw_output, list):
        raw_output = "".join(
            str(item)
            for item in raw_output
        )

    raw_output = str(raw_output).strip()

    print("\n[PLANNER RAW OUTPUT]")
    print(raw_output)

    search_plan: list[str] = []

    for line in raw_output.splitlines():
        query = line.strip()

        # Remove accidental bullets.
        query = query.lstrip("-•* ")

        # Remove accidental numbering such as "1. query".
        if len(query) > 2 and query[0].isdigit():
            parts = query.split(".", 1)

            if len(parts) == 2:
                query = parts[1].strip()

        if query:
            search_plan.append(query)

    # Remove duplicate queries while preserving order.
    deduplicated_plan: list[str] = []
    seen: set[str] = set()

    for query in search_plan:
        normalized = query.lower().strip()

        if normalized not in seen:
            deduplicated_plan.append(query)
            seen.add(normalized)

    search_plan = deduplicated_plan[:3]

    # Deterministically protect exact versions/numbers/IDs from planner
    # corruption.
    corrected_plan: list[str] = []

    for query in search_plan:
        corrected_query = _preserve_exact_tokens(
            query,
            state["question"],
        )

        if corrected_query:
            corrected_plan.append(
                corrected_query
            )

    search_plan = corrected_plan

    # Safety fallback.
    if not search_plan:
        search_plan = [
            state["question"],
            f"Kestrel Labs documentation {state['question']}",
        ]

    print("\n[PLANNER SEARCH PLAN]")

    for query in search_plan:
        print(f"- {query}")

    return {
        "search_plan": search_plan,
    }