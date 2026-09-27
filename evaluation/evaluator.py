from pathlib import Path
import fitz
import re
import json
from collections import Counter


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "data" / "uploaded_pdfs"

QUESTIONS_FILE = (
    BASE_DIR
    / "evaluation"
    / "generated_questions.json"
)

QUESTIONS_PER_DOCUMENT = 5


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "about", "after", "again", "against", "also",
    "among", "because", "before", "being", "between",
    "could", "during", "each", "from", "further",
    "have", "having", "into", "more", "most", "other",
    "over", "same", "such", "than", "that", "their",
    "there", "these", "they", "this", "those",
    "through", "under", "using", "used", "very",
    "were", "which", "while", "with", "would",
    "where", "when", "what", "whose", "will",
    "shall", "should", "then", "them", "some",
    "many", "only", "been", "being", "does",
    "doesnt", "research", "study", "paper",
    "article", "method", "methods", "results",
    "result", "figure", "table", "section",
    "proposed", "approach", "system",
    "based", "different", "following",
    "shown", "presented", "provide",
    "provides", "described", "discussed"
}


# ============================================================
# EXTRACT PDF TEXT
# ============================================================

def extract_documents():

    documents = []

    pdf_files = list(
        UPLOAD_DIR.glob("*.pdf")
    )

    if not pdf_files:

        print("No uploaded PDFs found.")

        return documents

    for pdf_path in pdf_files:

        try:

            doc = fitz.open(
                pdf_path
            )

            pages = []

            for page_number, page in enumerate(
                doc,
                start=1
            ):

                text = page.get_text(
                    "text"
                ).strip()

                if text:

                    pages.append(
                        {
                            "page": page_number,
                            "text": text
                        }
                    )

            doc.close()

            full_text = "\n".join(
                page["text"]
                for page in pages
            )

            if full_text.strip():

                documents.append(
                    {
                        "filename": pdf_path.name,
                        "text": full_text,
                        "pages": pages
                    }
                )

        except Exception as error:

            print(
                f"Could not read "
                f"{pdf_path.name}: {error}"
            )

    return documents


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# SPLIT SENTENCES
# ============================================================

def split_sentences(text):

    text = clean_text(
        text
    )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) < 70:
            continue

        if len(sentence) > 400:
            continue

        lower = sentence.lower()

        if "copyright" in lower:
            continue

        if "issn" in lower:
            continue

        if "www." in lower:
            continue

        cleaned.append(
            sentence
        )

    return cleaned


# ============================================================
# GET CONTENT WORDS
# ============================================================

def get_content_words(
    sentence
):

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z-]{3,}\b",
        sentence.lower()
    )

    words = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    return words


# ============================================================
# DOCUMENT TERM FREQUENCY
# ============================================================

def document_term_frequency(
    text
):

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z-]{3,}\b",
        text.lower()
    )

    words = [
        word
        for word in words
        if word not in STOP_WORDS
    ]

    return Counter(words)


# ============================================================
# FIND IMPORTANT SENTENCES
# ============================================================

def find_important_sentences(
    text
):

    sentences = split_sentences(
        text
    )

    frequency = document_term_frequency(
        text
    )

    scored = []

    for sentence in sentences:

        words = get_content_words(
            sentence
        )

        if not words:
            continue

        unique_words = set(words)

        score = 0

        for word in unique_words:

            score += frequency.get(
                word,
                0
            )

        # Prefer sentences with
        # multiple meaningful terms.

        score += len(
            unique_words
        ) * 2

        scored.append(
            (
                score,
                sentence
            )
        )

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        sentence
        for score, sentence
        in scored
    ]


# ============================================================
# BUILD SPECIFIC QUERY
# ============================================================

def build_question(
    sentence,
    question_type
):

    words = get_content_words(
        sentence
    )

    if len(words) < 2:

        return None

    # Remove duplicates while
    # preserving order.

    unique_words = []

    for word in words:

        if word not in unique_words:

            unique_words.append(
                word
            )

    # Use several terms from the
    # SAME sentence so the query
    # remains specific.

    selected_terms = unique_words[:4]

    topic = " ".join(
        selected_terms
    )


    if question_type == 0:

        return (
            f"What does the research "
            f"report about {topic}?"
        )


    if question_type == 1:

        return (
            f"How are {topic} "
            f"discussed in the research?"
        )


    if question_type == 2:

        return (
            f"What role do {topic} "
            f"play in the research?"
        )


    if question_type == 3:

        return (
            f"What findings are reported "
            f"about {topic}?"
        )


    return (
        f"What is the significance of "
        f"{topic} in the research?"
    )


# ============================================================
# GENERATE QUESTIONS
# ============================================================

def generate_questions(
    document
):

    filename = document["filename"]

    text = document["text"]

    sentences = find_important_sentences(
        text
    )

    questions = []

    used_queries = set()


    for index, sentence in enumerate(
        sentences
    ):

        if len(questions) >= QUESTIONS_PER_DOCUMENT:
            break

        question = build_question(
            sentence,
            index % 5
        )

        if not question:
            continue

        normalized = question.lower()

        if normalized in used_queries:
            continue

        used_queries.add(
            normalized
        )

        questions.append(
            {
                "question": question,
                "type": "content_specific",
                "document": filename,
                "reference_context":
                    sentence
            }
        )

    return questions


# ============================================================
# GENERATE ALL QUESTIONS
# ============================================================

def generate_evaluation_questions(
    documents
):

    all_questions = []

    for document in documents:

        questions = generate_questions(
            document
        )

        all_questions.extend(
            questions
        )

    return all_questions


# ============================================================
# SAVE QUESTIONS
# ============================================================

def save_questions(
    questions
):

    with open(
        QUESTIONS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            questions,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "AI RESEARCH ASSISTANT"
    )

    print(
        "CONTENT-SPECIFIC EVALUATION QUERY GENERATOR"
    )

    print("=" * 70)


    documents = extract_documents()


    print(
        f"\nUploaded PDFs found: "
        f"{len(documents)}"
    )


    if not documents:

        print(
            "\nPlease upload PDFs first."
        )

        return


    questions = (
        generate_evaluation_questions(
            documents
        )
    )


    print("\n")
    print("=" * 70)

    print(
        "GENERATED CONTENT-SPECIFIC QUESTIONS"
    )

    print("=" * 70)


    for index, item in enumerate(
        questions,
        start=1
    ):

        print(
            f"\n{index}. "
            f"{item['question']}"
        )

        print(
            f"   Document: "
            f"{item['document']}"
        )

        print(
            f"   Reference: "
            f"{item['reference_context'][:180]}..."
        )


    save_questions(
        questions
    )


    print("\n")
    print("=" * 70)

    print(
        f"✓ Generated "
        f"{len(questions)} questions."
    )

    print(
        "✓ Saved to:"
    )

    print(
        QUESTIONS_FILE
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()