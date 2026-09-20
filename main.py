from src.graph import build_graph


def main() -> None:
    graph = build_graph()

    print("=" * 60)
    print("Kestrel Research Assistant")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)

    conversation_history = []

    while True:
        question = input("\nYou: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("\nGoodbye.")
            break

        try:
            result = graph.invoke(
                {
                    "question": question,
                    "conversation_history": conversation_history,
                }
            )

            answer = result.get(
                "final_answer",
                "No final answer was produced.",
            )

            verdict = result.get(
                "verifier_verdict",
                "insufficient_evidence",
            )

            citations = result.get(
                "citations",
                [],
            )

            print("\nAssistant:")
            print(answer)

            print(f"\nVerifier verdict: {verdict}")

            if citations:
                print("\nSources:")

                for citation in citations:
                    print(
                        f"- {citation.get('chunk_id', 'unknown')}: "
                        f"{citation.get('title', 'Unknown document')}"
                    )

            conversation_history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            conversation_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

        except Exception as exc:
            print(
                f"\nWorkflow error: {exc}"
            )


if __name__ == "__main__":
    main()