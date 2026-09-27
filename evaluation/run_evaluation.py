from pathlib import Path
import json
import urllib.request
import urllib.error


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

QUESTIONS_FILE = (
    BASE_DIR
    / "evaluation"
    / "generated_questions.json"
)

RESULTS_FILE = (
    BASE_DIR
    / "evaluation"
    / "evaluation_results.json"
)

API_URL = "http://127.0.0.1:8000/ask"


# ============================================================
# LOAD QUESTIONS
# ============================================================

def load_questions():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# CALL RAG API
# ============================================================

def ask_rag(question):

    payload = {
        "question": question
    }

    data = json.dumps(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        API_URL,
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=180
        ) as response:

            return json.loads(
                response.read()
                .decode("utf-8")
            )

    except urllib.error.HTTPError as error:

        print(
            f"API ERROR: {error.code}"
        )

        try:

            print(
                error.read()
                .decode("utf-8")
            )

        except Exception:
            pass

        return None

    except Exception as error:

        print(
            "CONNECTION ERROR:",
            error
        )

        return None


# ============================================================
# GET SOURCE NAME
# ============================================================

def get_filename(source):

    if not isinstance(
        source,
        dict
    ):
        return ""

    return str(
        source.get(
            "filename",
            source.get(
                "source",
                ""
            )
        )
    )


# ============================================================
# EVALUATE ONE QUESTION
# ============================================================

def evaluate_question(
    item,
    number,
    total
):

    question = item.get(
        "question",
        ""
    )

    expected_document = item.get(
        "document",
        ""
    )


    print("\n")
    print("=" * 70)

    print(
        f"QUESTION {number}/{total}"
    )

    print("=" * 70)

    print(
        "Question:",
        question
    )

    print(
        "Expected:",
        expected_document
    )


    result = ask_rag(
        question
    )


    if result is None:

        return {
            "question": question,
            "expected_document":
                expected_document,
            "status": "api_error"
        }


    # --------------------------------------------------------
    # Show actual API fields
    # --------------------------------------------------------

    if number == 1:

        print("\nAPI RESPONSE FIELDS:")

        for key in result.keys():

            print(
                " -",
                key
            )


    # --------------------------------------------------------
    # Read RAG metrics
    # --------------------------------------------------------

    retrieved_chunks = result.get(
        "retrieved_chunks",
        0
    )

    unique_sources = result.get(
        "unique_sources",
        0
    )

    average_similarity = result.get(
        "avg_similarity",
        0
    )

    answer = result.get(
        "answer",
        ""
    )

    sources = result.get(
        "sources",
        []
    )


    if not isinstance(
        sources,
        list
    ):

        sources = []


    # --------------------------------------------------------
    # Retrieved documents
    # --------------------------------------------------------

    retrieved_documents = []


    for source in sources:

        filename = get_filename(
            source
        )

        if filename:

            if filename not in retrieved_documents:

                retrieved_documents.append(
                    filename
                )


    # --------------------------------------------------------
    # Expected document check
    # --------------------------------------------------------

    expected_found = False

    expected_lower = (
        expected_document
        .strip()
        .lower()
    )


    for filename in retrieved_documents:

        if (
            filename.strip().lower()
            == expected_lower
        ):

            expected_found = True

            break


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "Retrieved chunks:",
        retrieved_chunks
    )

    print(
        "Unique sources:",
        unique_sources
    )

    print(
        "Average similarity:",
        average_similarity
    )

    print(
        "Expected document found:",
        "YES"
        if expected_found
        else "NO"
    )


    print(
        "Retrieved documents:"
    )


    if retrieved_documents:

        for filename in retrieved_documents:

            print(
                "  -",
                filename
            )

    else:

        print(
            "  None"
        )


    return {
        "question": question,
        "type": item.get(
            "type",
            ""
        ),
        "topic_term": item.get(
            "topic_term",
            ""
        ),
        "expected_document":
            expected_document,
        "status": "success",
        "retrieved_chunks":
            retrieved_chunks,
        "unique_sources":
            unique_sources,
        "average_similarity":
            average_similarity,
        "expected_document_retrieved":
            expected_found,
        "retrieved_documents":
            retrieved_documents,
        "answer":
            answer,
        "sources":
            sources
    }


# ============================================================
# SUMMARY
# ============================================================

def create_summary(results):

    successful = [
        result
        for result in results
        if result.get("status")
        == "success"
    ]


    if not successful:

        return {
            "total_questions":
                len(results),
            "successful_questions":
                0,
            "document_retrieval_accuracy":
                0,
            "average_similarity":
                0,
            "average_retrieved_chunks":
                0
        }


    correct = sum(
        1
        for result in successful
        if result.get(
            "expected_document_retrieved",
            False
        )
    )


    similarity_values = []

    chunk_values = []


    for result in successful:

        try:

            similarity_values.append(
                float(
                    result.get(
                        "average_similarity",
                        0
                    )
                )
            )

        except Exception:
            pass


        try:

            chunk_values.append(
                float(
                    result.get(
                        "retrieved_chunks",
                        0
                    )
                )
            )

        except Exception:
            pass


    return {
        "total_questions":
            len(results),

        "successful_questions":
            len(successful),

        "document_retrieval_accuracy":
            round(
                correct / len(successful),
                4
            ),

        "average_similarity":
            round(
                sum(similarity_values)
                / len(similarity_values),
                4
            )
            if similarity_values
            else 0,

        "average_retrieved_chunks":
            round(
                sum(chunk_values)
                / len(chunk_values),
                2
            )
            if chunk_values
            else 0
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "AI RESEARCH ASSISTANT"
    )

    print(
        "AUTOMATIC RAG EVALUATION"
    )

    print("=" * 70)


    questions = load_questions()


    print(
        f"\nLoaded {len(questions)} questions."
    )


    results = []


    for number, item in enumerate(
        questions,
        start=1
    ):

        result = evaluate_question(
            item,
            number,
            len(questions)
        )

        results.append(
            result
        )


    summary = create_summary(
        results
    )


    output = {
        "summary": summary,
        "results": results
    }


    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )


    print("\n")
    print("=" * 70)

    print(
        "EVALUATION SUMMARY"
    )

    print("=" * 70)

    print(
        "Total questions:",
        summary[
            "total_questions"
        ]
    )

    print(
        "Successful questions:",
        summary[
            "successful_questions"
        ]
    )

    print(
        "Document retrieval accuracy:",
        summary[
            "document_retrieval_accuracy"
        ]
    )

    print(
        "Average similarity:",
        summary[
            "average_similarity"
        ]
    )

    print(
        "Average retrieved chunks:",
        summary[
            "average_retrieved_chunks"
        ]
    )


    print("\n✓ Evaluation completed.")

    print(
        "✓ Results saved to:"
    )

    print(
        RESULTS_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()