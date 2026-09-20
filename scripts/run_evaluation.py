import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.graph import build_graph
from src.utils.llm import get_llm


QUESTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "eval_questions.jsonl"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "eval_results.jsonl"
)

METRICS_FILE = (
    PROJECT_ROOT
    / "results"
    / "metrics_summary.json"
)


def load_questions():
    questions = []

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                questions.append(
                    json.loads(line)
                )

    return questions


def normalize_text(text):
    """
    Normalize text for lightweight answer comparison.
    """

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def extract_tokens(text):
    normalized = normalize_text(text)

    if not normalized:
        return set()

    return set(
        normalized.split()
    )


def calculate_token_overlap(
    predicted,
    expected,
):
    """
    Lightweight lexical overlap between the generated
    answer and expected answer.

    This is NOT an LLM judge.
    """

    predicted_tokens = extract_tokens(
        predicted
    )

    expected_tokens = extract_tokens(
        expected
    )

    if not expected_tokens:
        return 0.0

    overlap = (
        predicted_tokens
        & expected_tokens
    )

    return round(
        len(overlap)
        / len(expected_tokens),
        3,
    )


def calculate_retrieval_metrics(
    expected_chunk_ids,
    retrieved_chunk_ids,
):
    expected = set(
        expected_chunk_ids or []
    )

    retrieved = set(
        retrieved_chunk_ids or []
    )

    if not expected:
        return {
            "retrieval_recall": None,
            "retrieval_precision": None,
        }

    intersection = (
        expected
        & retrieved
    )

    recall = (
        len(intersection)
        / len(expected)
    )

    precision = (
        len(intersection)
        / len(retrieved)
        if retrieved
        else 0.0
    )

    return {
        "retrieval_recall": round(
            recall,
            3,
        ),
        "retrieval_precision": round(
            precision,
            3,
        ),
    }


def calculate_citation_precision(
    citations,
    expected_chunk_ids,
):
    """
    Measures how many cited chunks are part of the
    expected supporting evidence.
    """

    expected = set(
        expected_chunk_ids or []
    )

    cited = set()

    for citation in citations or []:

        if isinstance(
            citation,
            dict,
        ):

            chunk_id = citation.get(
                "chunk_id"
            )

            if chunk_id:
                cited.add(
                    chunk_id
                )

    if not cited:
        return 0.0

    correct = (
        cited
        & expected
    )

    return round(
        len(correct)
        / len(cited),
        3,
    )


def calculate_citation_recall(
    citations,
    expected_chunk_ids,
):
    """
    Measures how much of the expected evidence
    was explicitly cited.
    """

    expected = set(
        expected_chunk_ids or []
    )

    if not expected:
        return None

    cited = {
        citation.get("chunk_id")
        for citation in (
            citations or []
        )
        if isinstance(
            citation,
            dict,
        )
        and citation.get("chunk_id")
    }

    overlap = (
        expected
        & cited
    )

    return round(
        len(overlap)
        / len(expected),
        3,
    )


def evaluate_answer(
    question_item,
    answer,
):
    """
    Lightweight deterministic answer evaluation.

    For supported questions:
        lexical overlap against expected answer.

    For unsupported questions:
        checks whether the answer communicates
        insufficient evidence.
    """

    expected_answer = question_item.get(
        "expected_answer"
    )

    if expected_answer is None:

        normalized_answer = normalize_text(
            answer
        )

        insufficient_phrases = [
            "does not establish",
            "insufficient evidence",
            "not enough evidence",
            "documentation does not",
            "cannot determine",
            "not established",
            "no evidence",
        ]

        handled = any(
            phrase in normalized_answer
            for phrase in insufficient_phrases
        )

        return {
            "answer_match": 1.0
            if handled
            else 0.0,
            "unsupported_handled": handled,
        }

    overlap = calculate_token_overlap(
        answer,
        expected_answer,
    )

    return {
        "answer_match": overlap,
        "unsupported_handled": None,
    }


def calculate_question_scores(
    question_item,
    result,
):
    expected_chunk_ids = (
        question_item.get(
            "expected_chunk_ids",
            [],
        )
    )

    retrieved_chunk_ids = (
        result.get(
            "retrieved_chunk_ids",
            [],
        )
    )

    citations = result.get(
        "citations",
        [],
    )

    retrieval_metrics = (
        calculate_retrieval_metrics(
            expected_chunk_ids,
            retrieved_chunk_ids,
        )
    )

    citation_precision = (
        calculate_citation_precision(
            citations,
            expected_chunk_ids,
        )
    )

    citation_recall = (
        calculate_citation_recall(
            citations,
            expected_chunk_ids,
        )
    )

    answer_metrics = evaluate_answer(
        question_item,
        result.get(
            "answer",
            "",
        ),
    )

    scores = {
        **retrieval_metrics,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        **answer_metrics,
    }

    return scores


def calculate_average(
    values,
):
    values = [
        value
        for value in values
        if value is not None
    ]

    if not values:
        return None

    return round(
        sum(values)
        / len(values),
        3,
    )


def calculate_aggregate_metrics(
    questions,
    results,
):
    question_lookup = {
        item["question_id"]: item
        for item in questions
    }

    metric_values = defaultdict(
        list
    )

    type_metrics = defaultdict(
        lambda: defaultdict(list)
    )

    for result in results:

        question_id = result.get(
            "question_id"
        )

        question_item = question_lookup.get(
            question_id
        )

        if not question_item:
            continue

        scores = result.get(
            "scores",
            {},
        )

        question_type = question_item.get(
            "type",
            "unknown",
        )

        for metric, value in scores.items():

            if isinstance(
                value,
                (int, float),
            ):

                metric_values[
                    metric
                ].append(value)

                type_metrics[
                    question_type
                ][metric].append(value)

    overall = {
        metric: calculate_average(
            values
        )
        for metric, values
        in metric_values.items()
    }

    by_type = {}

    for question_type, metrics in type_metrics.items():

        by_type[question_type] = {
            metric: calculate_average(
                values
            )
            for metric, values
            in metrics.items()
        }

    return {
        "overall": overall,
        "by_question_type": by_type,
    }


def get_model_info():
    """
    Read configured model names without making
    additional LLM calls.
    """

    llm = get_llm()

    generation_model = getattr(
        llm,
        "model_name",
        None,
    )

    if not generation_model:
        generation_model = getattr(
            llm,
            "model",
            "unknown",
        )

    return {
        "generation_model": generation_model,
        "embedding_model": (
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
    }


def run_evaluation():

    questions = load_questions()

    print("=" * 70)
    print(
        "KESTREL RESEARCH ASSISTANT - EVALUATION"
    )
    print("=" * 70)

    print(
        f"Questions: {len(questions)}"
    )

    print()

    app = build_graph()

    results = []

    conversations = {}

    total_start = time.perf_counter()

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question_id = item[
            "question_id"
        ]

        question = item[
            "question"
        ]

        conversation_id = item[
            "conversation_id"
        ]

        print(
            f"[{index}/{len(questions)}] "
            f"{question_id}: {question}"
        )

        history = conversations.get(
            conversation_id,
            [],
        )

        state = {
            "question": question,
            "conversation_history": history,
        }

        start_time = time.perf_counter()

        try:

            result = app.invoke(
                state
            )

            latency = (
                time.perf_counter()
                - start_time
            )

            answer = result.get(
                "final_answer",
                "",
            )

            citations = result.get(
                "citations",
                [],
            )

            retrieved_chunks = result.get(
                "retrieved_chunks",
                [],
            )

            retrieved_chunk_ids = [
                chunk["chunk_id"]
                for chunk in retrieved_chunks
                if "chunk_id" in chunk
            ]

            verifier_verdict = result.get(
                "verifier_verdict",
                "insufficient_evidence",
            )

            output = {
                "question_id": question_id,
                "answer": answer,
                "citations": citations,
                "retrieved_chunk_ids": (
                    retrieved_chunk_ids
                ),
                "verifier_verdict": (
                    verifier_verdict
                ),
                "scores": {},
                "latency_seconds": round(
                    latency,
                    3,
                ),
                "langsmith_run_url": None,
            }

            output["scores"] = (
                calculate_question_scores(
                    item,
                    output,
                )
            )

            # IMPORTANT:
            # A successful graph execution must explicitly receive
            # execution_error = 0. Otherwise aggregate_metrics()
            # averages only failed records and reports 1.0 whenever
            # all recorded failures have value 1.
            output["scores"]["execution_error"] = 0

            # Save current turn for follow-up questions.
            history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            conversations[
                conversation_id
            ] = history

        except Exception as e:

            latency = (
                time.perf_counter()
                - start_time
            )

            print(
                f"  ERROR: {e}"
            )

            output = {
                "question_id": question_id,
                "answer": "",
                "citations": [],
                "retrieved_chunk_ids": [],
                "verifier_verdict": (
                    "insufficient_evidence"
                ),
                "scores": {
                    "execution_error": 1,
                },
                "latency_seconds": round(
                    latency,
                    3,
                ),
                "langsmith_run_url": None,
                "error": str(e),
            }

        results.append(
            output
        )

        print(
            f"  Verdict: "
            f"{output['verifier_verdict']}"
        )

        print(
            f"  Latency: "
            f"{output['latency_seconds']}s"
        )

        print(
            f"  Scores: "
            f"{output['scores']}"
        )

        print()

    total_wall_clock = (
        time.perf_counter()
        - total_start
    )

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        for result in results:

            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    aggregate_metrics = (
        calculate_aggregate_metrics(
            questions,
            results,
        )
    )

    model_info = get_model_info()

    successful_results = [
        result
        for result in results
        if result.get(
            "scores",
            {},
        ).get(
            "execution_error"
        ) == 0
    ]

    failed_results = [
        result
        for result in results
        if result.get(
            "scores",
            {},
        ).get(
            "execution_error"
        ) == 1
    ]

    metrics_summary = {
        "evaluation": {
            "total_questions": len(
                questions
            ),
            "successful_questions": len(
                successful_results
            ),
            "failed_questions": len(
                failed_results
            ),
        },
        "metrics": aggregate_metrics[
            "overall"
        ],
        "breakdown_by_question_type": (
            aggregate_metrics[
                "by_question_type"
            ]
        ),
        "total_wall_clock_seconds": round(
            total_wall_clock,
            3,
        ),
        "total_latency_seconds": round(
            sum(
                result.get(
                    "latency_seconds",
                    0,
                )
                for result in results
            ),
            3,
        ),
        "models": model_info,
        "notes": [
            "Answer match is a lightweight lexical "
            "comparison against expected_answer.",
            "Retrieval metrics use expected_chunk_ids "
            "as the evaluation reference.",
            "Citation precision measures the fraction "
            "of cited chunks that match expected evidence.",
            "Execution error is 1 for a failed graph "
            "execution and 0 for a successful execution.",
            "LangSmith run URLs will be added in the "
            "observability integration step.",
        ],
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics_summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        "Results written to:"
    )

    print(
        RESULTS_FILE
    )

    print()

    print(
        "Metrics written to:"
    )

    print(
        METRICS_FILE
    )

    print()

    print(
        "Overall metrics:"
    )

    for metric, value in (
        aggregate_metrics[
            "overall"
        ].items()
    ):

        print(
            f"  {metric}: {value}"
        )

    print()

    print(
        f"Total wall-clock time: "
        f"{round(total_wall_clock, 3)}s"
    )


if __name__ == "__main__":
    run_evaluation()