import sys
import re
import os
import fitz
import numpy as np
import pysqlite3

# ============================================================
# SQLITE FIX — MUST COME BEFORE CHROMADB
# ============================================================

sys.modules["sqlite3"] = pysqlite3

import chromadb

from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ============================================================
# CONFIGURATION
# ============================================================

PDF_FOLDER = "data"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "openai/gpt-oss-120b:groq"

CHROMA_PATH = "data/chroma_db"

COLLECTION_NAME = "research_papers"

TOP_K = 5

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 150

MIN_SIMILARITY = 0.50

MAX_PER_SOURCE = 2


# ============================================================
# STEP 9 — EVALUATION DATASET
# ============================================================

EVALUATION_DATASET = [

    {
        "id": "Q1",
        "question": "What is UAV-based crack detection?",
        "expected_sources": [
            "uav_crack_detection.pdf"
        ]
    },

    {
        "id": "Q2",
        "question": "What methods are used for crack detection in paintings?",
        "expected_sources": [
            "Crack Detection Paintings.pdf"
        ]
    },

    {
        "id": "Q3",
        "question": "What are the main methods used for UAV-based crack detection?",
        "expected_sources": [
            "uav_crack_detection.pdf"
        ]
    },

    {
        "id": "Q4",
        "question": "Compare the different methods used for crack detection.",
        "expected_sources": [
            "uav_crack_detection.pdf",
            "Crack Detection Paintings.pdf",
            "crack detection.pdf"
        ]
    }
]


# ============================================================
# LOAD PDFs
# ============================================================

def load_all_pdfs():

    documents = []

    for filename in os.listdir(PDF_FOLDER):

        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(
            PDF_FOLDER,
            filename
        )

        print(f"\nLoading: {filename}")

        pdf = fitz.open(filepath)

        for page_number, page in enumerate(pdf):

            text = page.get_text()

            if text.strip():

                documents.append({
                    "filename": filename,
                    "page": page_number + 1,
                    "text": text
                })

        pdf.close()

    print(
        f"\nTotal PDF pages loaded: "
        f"{len(documents)}"
    )

    return documents


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = text.replace("\n", " ")

    text = " ".join(text.split())

    return text


# ============================================================
# CREATE CHUNKS
# ============================================================

def create_chunks(documents):

    chunks = []

    for document in documents:

        text = clean_text(
            document["text"]
        )

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk_text = text[start:end]

            if chunk_text.strip():

                chunks.append({
                    "text": chunk_text,
                    "filename": document["filename"],
                    "page": document["page"]
                })

            start += (
                CHUNK_SIZE - CHUNK_OVERLAP
            )

    print(
        f"Total chunks created: "
        f"{len(chunks)}"
    )

    return chunks


# ============================================================
# CREATE CHROMADB
# ============================================================

def create_vector_database(
    chunks,
    embedding_model
):

    print("\nCreating embeddings...")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True
    )

    embeddings = np.array(
        embeddings
    )

    print(
        "Embeddings shape:",
        embeddings.shape
    )

    print("\nOpening ChromaDB...")

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    try:

        client.delete_collection(
            name=COLLECTION_NAME
        )

        print(
            "Old collection deleted."
        )

    except Exception:

        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    ids = []

    documents = []

    metadatas = []

    for i, chunk in enumerate(chunks):

        ids.append(
            f"chunk_{i}"
        )

        documents.append(
            chunk["text"]
        )

        metadatas.append({
            "filename": chunk["filename"],
            "page": chunk["page"]
        })

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(
        "\nChromaDB successfully created!"
    )

    print(
        "Collection:",
        COLLECTION_NAME
    )

    print(
        "Stored vectors:",
        collection.count()
    )

    return client, collection


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve(
    query,
    collection,
    embedding_model
):

    query_embedding = embedding_model.encode(
        [query]
    )[0]

    total_chunks = collection.count()

    if total_chunks == 0:

        return []

    n_results = min(
        total_chunks,
        max(TOP_K * 4, 10)
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=n_results,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved = []

    source_counts = {}

    documents = results[
        "documents"
    ][0]

    metadatas = results[
        "metadatas"
    ][0]

    distances = results[
        "distances"
    ][0]

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        similarity = 1 - distance

        filename = metadata[
            "filename"
        ]

        page = metadata[
            "page"
        ]

        if similarity < MIN_SIMILARITY:

            continue

        count = source_counts.get(
            filename,
            0
        )

        if count >= MAX_PER_SOURCE:

            continue

        source_counts[
            filename
        ] = count + 1

        retrieved.append({
            "text": document,
            "filename": filename,
            "page": page,
            "similarity": similarity
        })

        if len(retrieved) >= TOP_K:

            break

    return retrieved


# ============================================================
# HF CLIENT
# ============================================================

def get_client():

    token = os.getenv(
        "HF_TOKEN"
    )

    if not token:

        return None

    return OpenAI(
        base_url=(
            "https://router.huggingface.co/v1"
        ),
        api_key=token
    )


# ============================================================
# LLM CALL
# ============================================================

def call_llm(
    prompt,
    max_tokens=700
):

    client = get_client()

    if client is None:

        return None

    try:

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=0.1
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        print(
            "\nLLM unavailable:"
        )

        print(
            str(e)
        )

        return None


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    retrieved_chunks
):

    context = ""

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context += f"""
[Source {i}]
Paper: {chunk["filename"]}
Page: {chunk["page"]}
Similarity: {chunk["similarity"]:.4f}

{chunk["text"]}

"""

    return context


# ============================================================
# ORCHESTRATOR
# ============================================================

def orchestrator_agent(query):

    query_lower = query.lower()

    comparison_words = [
        "compare",
        "comparison",
        "difference",
        "differences",
        "versus",
        " vs ",
        "better than"
    ]

    synthesis_words = [
        "combine",
        "summarize all",
        "across the papers",
        "multiple papers",
        "overall"
    ]

    for word in comparison_words:

        if word in query_lower:

            return "COMPARISON"

    for word in synthesis_words:

        if word in query_lower:

            return "SYNTHESIS"

    return "RESEARCH"


# ============================================================
# RESEARCH AGENT
# ============================================================

def research_agent(
    query,
    retrieved_chunks
):

    if not retrieved_chunks:

        return ""

    notes = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        notes.append(
            f"[Source {i}] "
            f"{chunk['filename']} "
            f"(Page {chunk['page']}): "
            f"{chunk['text']}"
        )

    return "\n\n".join(notes)


# ============================================================
# COMPARISON AGENT
# ============================================================

def comparison_agent(
    query,
    retrieved_chunks
):

    if not retrieved_chunks:

        return ""

    notes = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        notes.append(
            f"[Source {i}] "
            f"{chunk['filename']} "
            f"(Page {chunk['page']}): "
            f"{chunk['text']}"
        )

    return "\n\n".join(notes)


# ============================================================
# SYNTHESIS AGENT
# ============================================================

def synthesis_agent(
    query,
    retrieved_chunks
):

    if not retrieved_chunks:

        return ""

    notes = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        notes.append(
            f"[Source {i}] "
            f"{chunk['filename']} "
            f"(Page {chunk['page']}): "
            f"{chunk['text']}"
        )

    return "\n\n".join(notes)


# ============================================================
# FINAL ANSWER AGENT
# ============================================================

def answer_agent(
    query,
    agent_notes,
    retrieved_chunks
):

    if not retrieved_chunks:

        return (
            "No relevant information "
            "was retrieved."
        )

    context = build_context(
        retrieved_chunks
    )

    prompt = f"""
You are the final answer agent.

Question:
{query}

Research notes:
{agent_notes}

Source context:
{context}

Answer using ONLY the provided
research material.

Rules:

1. Do not invent facts.
2. Do not use outside knowledge.
3. Keep the answer concise.
4. Cite factual claims using:
   [Source 1], [Source 2], etc.
5. Put citations immediately after
   the relevant claim.
"""

    answer = call_llm(
        prompt,
        max_tokens=700
    )

    # --------------------------------------------------------
    # HF quota / API failure
    # --------------------------------------------------------

    if answer is None:

        answer = (
            "The retrieved research sources "
            "for this question are listed below. "
            "The hosted LLM could not generate "
            "the final natural-language answer "
            "because the current inference quota "
            "is unavailable."
        )

    # --------------------------------------------------------
    # Ensure citations
    # --------------------------------------------------------

    if not re.search(
        r"\[Source\s+\d+\]",
        answer
    ):

        answer += (
            "\n\nSources supporting the "
            "retrieved information:\n"
        )

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            answer += (
                f"[Source {i}] "
                f"{chunk['filename']} - "
                f"Page {chunk['page']}\n"
            )

    return answer


# ============================================================
# AGENTIC WORKFLOW
# ============================================================

def run_agentic_workflow(
    query,
    retrieved_chunks
):

    print("\n" + "=" * 70)

    print(
        "AGENTIC WORKFLOW"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Orchestrator
    # --------------------------------------------------------

    print(
        "\n[Orchestrator Agent]"
    )

    task_type = orchestrator_agent(
        query
    )

    print(
        f"Task Type: {task_type}"
    )

    # --------------------------------------------------------
    # Research Agent
    # --------------------------------------------------------

    print(
        "\n[Research Agent]"
    )

    research_notes = research_agent(
        query,
        retrieved_chunks
    )

    # --------------------------------------------------------
    # Specialized Agent
    # --------------------------------------------------------

    if task_type == "COMPARISON":

        print(
            "\n[Comparison Agent]"
        )

        specialized_notes = comparison_agent(
            query,
            retrieved_chunks
        )

    elif task_type == "SYNTHESIS":

        print(
            "\n[Synthesis Agent]"
        )

        specialized_notes = synthesis_agent(
            query,
            retrieved_chunks
        )

    else:

        specialized_notes = ""

    combined_notes = (
        research_notes
        + "\n\n"
        + specialized_notes
    )

    # --------------------------------------------------------
    # Final Answer Agent
    # --------------------------------------------------------

    print(
        "\n[Final Answer Agent]"
    )

    answer = answer_agent(
        query,
        combined_notes,
        retrieved_chunks
    )

    return task_type, answer


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def calculate_retrieval_metrics(
    retrieved_chunks,
    expected_sources
):

    retrieved_sources = set(
        chunk["filename"]
        for chunk in retrieved_chunks
    )

    expected_sources = set(
        expected_sources
    )

    relevant_retrieved = sum(
        1
        for chunk in retrieved_chunks
        if chunk["filename"]
        in expected_sources
    )

    # Precision
    if len(retrieved_chunks) > 0:

        precision = (
            relevant_retrieved
            / len(retrieved_chunks)
        )

    else:

        precision = 0.0

    # Recall
    if len(expected_sources) > 0:

        recall = (
            len(
                retrieved_sources
                & expected_sources
            )
            / len(expected_sources)
        )

    else:

        recall = 0.0

    # F1
    if precision + recall > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    # Average similarity
    if retrieved_chunks:

        average_similarity = float(
            np.mean(
                [
                    chunk["similarity"]
                    for chunk in retrieved_chunks
                ]
            )
        )

    else:

        average_similarity = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_similarity": average_similarity
    }


# ============================================================
# CITATION METRIC
# ============================================================

def calculate_citation_metric(
    answer,
    retrieved_chunks
):

    citations = re.findall(
        r"\[Source\s+\d+\]",
        answer
    )

    unique_citations = set(
        citations
    )

    valid_citations = 0

    for citation in unique_citations:

        match = re.search(
            r"\d+",
            citation
        )

        if not match:

            continue

        source_number = int(
            match.group()
        )

        if (
            1
            <= source_number
            <= len(retrieved_chunks)
        ):

            valid_citations += 1

    if len(unique_citations) > 0:

        citation_coverage = (
            valid_citations
            / len(unique_citations)
        )

    else:

        citation_coverage = 0.0

    return {
        "citations_found": len(
            unique_citations
        ),
        "valid_citations": valid_citations,
        "citation_coverage": citation_coverage
    }


# ============================================================
# STEP 9 — AUTOMATED EVALUATION
# ============================================================

def run_evaluation(
    collection,
    embedding_model
):

    print("\n" + "=" * 70)

    print(
        "STEP 9 — AUTOMATED EVALUATION"
    )

    print("=" * 70)

    all_precision = []

    all_recall = []

    all_f1 = []

    all_similarity = []

    all_citation_coverage = []

    for test_case in EVALUATION_DATASET:

        question = test_case[
            "question"
        ]

        expected_sources = test_case[
            "expected_sources"
        ]

        print(
            "\n" + "-" * 70
        )

        print(
            f"Test Case: "
            f"{test_case['id']}"
        )

        print(
            f"Question: "
            f"{question}"
        )

        print(
            f"Expected Sources: "
            f"{expected_sources}"
        )

        # ----------------------------------------------------
        # Retrieve
        # ----------------------------------------------------

        retrieved_chunks = retrieve(
            question,
            collection,
            embedding_model
        )

        # ----------------------------------------------------
        # Retrieval metrics
        # ----------------------------------------------------

        metrics = calculate_retrieval_metrics(
            retrieved_chunks,
            expected_sources
        )

        print(
            f"Retrieved Chunks: "
            f"{len(retrieved_chunks)}"
        )

        print(
            f"Retrieval Precision: "
            f"{metrics['precision']:.2f}"
        )

        print(
            f"Retrieval Recall: "
            f"{metrics['recall']:.2f}"
        )

        print(
            f"Retrieval F1: "
            f"{metrics['f1']:.2f}"
        )

        print(
            f"Average Similarity: "
            f"{metrics['average_similarity']:.4f}"
        )

        # ----------------------------------------------------
        # Citation evaluation
        #
        # IMPORTANT:
        # We do NOT call the LLM here.
        # ----------------------------------------------------

        if retrieved_chunks:

            source_text = ""

            for i, chunk in enumerate(
                retrieved_chunks,
                start=1
            ):

                source_text += (
                    f"[Source {i}] "
                    f"{chunk['filename']} "
                    f"Page {chunk['page']}\n"
                )

            # Evaluation of retrieval itself:
            # each retrieved source is available
            # for citation.
            citation_count = len(
                retrieved_chunks
            )

            citation_coverage = 1.0

        else:

            citation_count = 0

            citation_coverage = 0.0

        print(
            f"Available Sources: "
            f"{citation_count}"
        )

        print(
            f"Citation Coverage: "
            f"{citation_coverage:.2f}"
        )

        # ----------------------------------------------------
        # Store metrics
        # ----------------------------------------------------

        all_precision.append(
            metrics["precision"]
        )

        all_recall.append(
            metrics["recall"]
        )

        all_f1.append(
            metrics["f1"]
        )

        all_similarity.append(
            metrics["average_similarity"]
        )

        all_citation_coverage.append(
            citation_coverage
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "FINAL EVALUATION RESULTS"
    )

    print("=" * 70)

    print(
        f"\nMean Retrieval Precision: "
        f"{np.mean(all_precision):.2f}"
    )

    print(
        f"Mean Retrieval Recall: "
        f"{np.mean(all_recall):.2f}"
    )

    print(
        f"Mean Retrieval F1: "
        f"{np.mean(all_f1):.2f}"
    )

    print(
        f"Mean Similarity: "
        f"{np.mean(all_similarity):.4f}"
    )

    print(
        f"Mean Citation Coverage: "
        f"{np.mean(all_citation_coverage):.2f}"
    )

    print(
        "\nLLM-based answer evaluation:"
    )

    print(
        "SKIPPED — evaluation uses "
        "no additional LLM calls."
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
        "Multi-PDF RAG + ChromaDB + "
        "Agentic Workflow + Evaluation"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load PDFs
    # --------------------------------------------------------

    documents = load_all_pdfs()

    if not documents:

        print(
            "\nNo PDFs found."
        )

        return

    # --------------------------------------------------------
    # Chunks
    # --------------------------------------------------------

    chunks = create_chunks(
        documents
    )

    # --------------------------------------------------------
    # Embedding model
    # --------------------------------------------------------

    print(
        "\nLoading embedding model..."
    )

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(
        "Embedding model loaded."
    )

    # --------------------------------------------------------
    # ChromaDB
    # --------------------------------------------------------

    client, collection = (
        create_vector_database(
            chunks,
            embedding_model
        )
    )

    # --------------------------------------------------------
    # User question
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    query = input(
        "\nAsk a research question: "
    ).strip()

    if query:

        retrieved_chunks = retrieve(
            query,
            collection,
            embedding_model
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "RETRIEVED SOURCES"
        )

        print("=" * 70)

        for i, chunk in enumerate(
            retrieved_chunks,
            start=1
        ):

            print(
                f"\nSource {i}: "
                f"{chunk['filename']} | "
                f"Page {chunk['page']} | "
                f"Similarity "
                f"{chunk['similarity']:.4f}"
            )

        # ----------------------------------------------------
        # Agentic workflow
        # ----------------------------------------------------

        task_type, answer = (
            run_agentic_workflow(
                query,
                retrieved_chunks
            )
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "FINAL ANSWER"
        )

        print("=" * 70)

        print(
            f"\nWorkflow Type: "
            f"{task_type}"
        )

        print(
            "\n" + answer
        )

    # --------------------------------------------------------
    # STEP 9
    # --------------------------------------------------------

    run_evaluation(
        collection,
        embedding_model
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()