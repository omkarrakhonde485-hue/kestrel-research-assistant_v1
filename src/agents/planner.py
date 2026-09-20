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
7. Preserve exact:
   - numbers
   - dates
   - version numbers
   - IDs
   - product names
   - plan names
   - technical terms
8. If the question asks about an incident, separate queries may target:
   - what happened
   - root cause
   - resolution/fix
9. If the question asks about a product or limit, target:
   - the specification
   - the relevant pricing/plan documentation
   - release notes if version history matters
10. If the question asks about conflicting information, create queries
    that retrieve the sources likely to contain each side of the conflict.
11. Do not answer the user's question.
12. Return ONLY the retrieval queries, one per line.
13. Do not number the queries.
14. Do not use bullets.
15. Do not add explanations.

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

    raw_output = response.content.strip()

    print("\n[PLANNER RAW OUTPUT]")
    print(raw_output)

    search_plan = []

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
    deduplicated_plan = []

    seen = set()

    for query in search_plan:
        normalized = query.lower().strip()

        if normalized not in seen:
            deduplicated_plan.append(query)
            seen.add(normalized)

    search_plan = deduplicated_plan[:3]

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