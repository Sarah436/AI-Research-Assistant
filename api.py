import sys
import os
import re
import math
import shutil
from pathlib import Path
from typing import List

# ============================================================
# SQLITE FIX FOR CHROMADB
# ============================================================

import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import chromadb
import fitz

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = DATA_DIR / "uploaded_pdfs"
CHROMA_DIR = DATA_DIR / "chroma_db"

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

COLLECTION_NAME = "research_papers"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 6
MAX_PER_SOURCE = 3

RETRIEVAL_THRESHOLD = 0.40

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AI Research Assistant",
    version="2.0",
    description="Multi-PDF RAG Research Assistant"
)

# ============================================================
# EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")

# ============================================================
# CHROMA
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "hnsw:space": "cosine"
    }
)

# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "what", "which", "who", "when", "where", "why",
    "how", "does", "do", "did", "can", "could",
    "would", "should", "will", "about", "for",
    "from", "with", "into", "and", "or", "of",
    "to", "in", "on", "by", "this", "that",
    "these", "those", "it", "its", "their",
    "there", "than", "used", "use", "using",
    "main", "methods", "method", "paper",
    "research", "study", "system", "approach"
}

# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str

# ============================================================
# TEXT HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str):
    text = normalize_text(text)

    words = re.findall(
        r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*",
        text
    )

    return [
        w for w in words
        if w not in STOPWORDS and len(w) > 1
    ]


def get_phrases(text: str):
    text = normalize_text(text)

    phrases = []

    # Quoted phrases
    quoted = re.findall(r'"([^"]+)"', text)

    for phrase in quoted:
        if len(phrase.split()) >= 2:
            phrases.append(phrase)

    # Important multi-word concepts
    words = tokenize(text)

    for i in range(len(words) - 1):
        phrase = words[i] + " " + words[i + 1]
        phrases.append(phrase)

    return list(dict.fromkeys(phrases))


def safe_filename(filename: str) -> str:
    filename = os.path.basename(filename)

    filename = re.sub(
        r"[^a-zA-Z0-9._()\- ]",
        "_",
        filename
    )

    return filename


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_pages(pdf_path: Path):

    pages = []

    try:
        doc = fitz.open(pdf_path)

        for page_number, page in enumerate(doc, start=1):

            text = page.get_text("text")

            if text and text.strip():

                pages.append({
                    "page": page_number,
                    "text": text.strip()
                })

        doc.close()

    except Exception as e:
        print("PDF extraction error:", e)

    return pages


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(text: str):

    text = re.sub(r"\s+", " ", text).strip()

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# DOCUMENT LIST
# ============================================================

def get_uploaded_documents():

    documents = []

    for pdf in sorted(
        UPLOAD_DIR.glob("*.pdf"),
        key=lambda x: x.name.lower()
    ):

        documents.append({
            "filename": pdf.name,
            "status": "Indexed"
        })

    return documents


# ============================================================
# ADD PDF TO CHROMA
# ============================================================

def index_pdf(pdf_path: Path):

    pages = extract_pdf_pages(pdf_path)

    if not pages:
        raise ValueError(
            "No readable text found in PDF."
        )

    ids = []
    documents = []
    metadatas = []

    chunk_counter = 0

    for page_data in pages:

        page_number = page_data["page"]
        page_text = page_data["text"]

        chunks = chunk_text(page_text)

        for chunk in chunks:

            chunk_counter += 1

            chunk_id = (
                f"{pdf_path.name}"
                f"__page_{page_number}"
                f"__chunk_{chunk_counter}"
            )

            ids.append(chunk_id)

            documents.append(chunk)

            metadatas.append({
                "filename": pdf_path.name,
                "page": page_number,
                "chunk_id": chunk_counter
            })

    if not documents:
        raise ValueError(
            "No text chunks generated."
        )

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return {
        "filename": pdf_path.name,
        "chunks_added": len(documents)
    }


# ============================================================
# GLOBAL IDF
# ============================================================

def calculate_global_idf():

    try:
        total = collection.count()

        if total == 0:
            return {}

        result = collection.get(
            include=["documents"]
        )

        docs = result.get("documents", [])

        if not docs:
            return {}

        document_frequency = {}

        total_docs = len(docs)

        for doc in docs:

            terms = set(tokenize(doc))

            for term in terms:

                document_frequency[term] = (
                    document_frequency.get(term, 0) + 1
                )

        idf = {}

        for term, frequency in document_frequency.items():

            idf[term] = math.log(
                (total_docs + 1) /
                (frequency + 1)
            ) + 1

        return idf

    except Exception as e:

        print("IDF error:", e)

        return {}


# ============================================================
# ENTITY DETECTION
# ============================================================

def detect_entity_phrases(question: str):

    entities = []

    # Capitalized multi-word names
    matches = re.findall(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        question
    )

    for match in matches:
        entities.append(
            normalize_text(match)
        )

    # Single capitalized terms
    for word in question.split():

        cleaned = re.sub(
            r"[^A-Za-z0-9\-]",
            "",
            word
        )

        if (
            len(cleaned) > 2
            and cleaned[:1].isupper()
        ):
            entities.append(
                normalize_text(cleaned)
            )

    return list(dict.fromkeys(entities))


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(question: str):

    total = collection.count()

    if total == 0:
        return {
            "chunks": [],
            "related_documents": [],
            "unrelated_documents": []
        }

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0].tolist()

    # Retrieve a broad candidate set
    candidate_count = min(
        max(TOP_K * 10, 30),
        total
    )

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_count,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    raw_documents = result.get("documents", [[]])[0]
    raw_metadatas = result.get("metadatas", [[]])[0]
    raw_distances = result.get("distances", [[]])[0]

    query_terms = tokenize(question)

    idf = calculate_global_idf()

    # High-information query terms
    if query_terms:

        weights = {
            term: idf.get(term, 1.0)
            for term in query_terms
        }

        mean_weight = sum(
            weights.values()
        ) / len(weights)

        specific_terms = {
            term
            for term, weight in weights.items()
            if weight >= mean_weight * 1.10
        }

    else:

        specific_terms = set()

    entity_phrases = detect_entity_phrases(
        question
    )

    print("\n================================================")
    print("QUERY:", question)
    print("ENTITY PHRASES:", entity_phrases)
    print("SPECIFIC TERMS:", list(specific_terms))
    print("================================================")

    candidates = []

    for i in range(len(raw_documents)):

        text = raw_documents[i]

        metadata = raw_metadatas[i] or {}

        filename = metadata.get(
            "filename",
            "Unknown"
        )

        page = metadata.get(
            "page",
            0
        )

        distance = raw_distances[i]

        semantic_similarity = max(
            0.0,
            1.0 - float(distance)
        )

        text_terms = set(
            tokenize(text)
        )

        # --------------------------------------------
        # Weighted term coverage
        # --------------------------------------------

        weighted_total = sum(
            idf.get(term, 1.0)
            for term in query_terms
        )

        weighted_matched = sum(
            idf.get(term, 1.0)
            for term in query_terms
            if term in text_terms
        )

        if weighted_total > 0:
            weighted_coverage = (
                weighted_matched /
                weighted_total
            )
        else:
            weighted_coverage = 0.0

        # --------------------------------------------
        # Specific term coverage
        # --------------------------------------------

        if specific_terms:

            matched_specific = [
                term
                for term in specific_terms
                if term in text_terms
            ]

            specific_coverage = (
                len(matched_specific) /
                len(specific_terms)
            )

        else:

            matched_specific = []
            specific_coverage = 0.0

        # --------------------------------------------
        # Phrase coverage
        # --------------------------------------------

        normalized_chunk = normalize_text(
            text
        )

        matched_phrases = 0

        if entity_phrases:

            for phrase in entity_phrases:

                if phrase in normalized_chunk:
                    matched_phrases += 1

        phrase_coverage = (
            matched_phrases /
            len(entity_phrases)
            if entity_phrases
            else 0.0
        )

        # --------------------------------------------
        # Entity coverage
        # --------------------------------------------

        entity_matches = []

        for entity in entity_phrases:

            entity_words = entity.split()

            if all(
                word in text_terms
                for word in entity_words
            ):
                entity_matches.append(entity)

        entity_coverage = (
            len(entity_matches) /
            len(entity_phrases)
            if entity_phrases
            else 0.0
        )

        # --------------------------------------------
        # Hybrid score
        # --------------------------------------------

        hybrid_score = (
            0.50 * semantic_similarity
            + 0.20 * weighted_coverage
            + 0.20 * specific_coverage
            + 0.05 * phrase_coverage
            + 0.05 * entity_coverage
        )

        # Specific-term penalty
        if specific_terms:

            if not matched_specific:
                hybrid_score *= 0.70

        # Entity penalty
        if entity_phrases:

            if entity_coverage == 0:
                hybrid_score *= 0.75

        candidates.append({

            "filename": filename,

            "page": page,

            "text": text,

            "semantic_similarity":
                semantic_similarity,

            "weighted_coverage":
                weighted_coverage,

            "specific_coverage":
                specific_coverage,

            "phrase_coverage":
                phrase_coverage,

            "entity_coverage":
                entity_coverage,

            "score":
                hybrid_score
        })

    # ========================================================
    # ENTITY HARD FILTER
    # ========================================================

    # If question contains a clear entity phrase,
    # prefer chunks containing that entity.
    if entity_phrases:

        entity_candidates = [
            item
            for item in candidates
            if item["entity_coverage"] > 0
        ]

        if entity_candidates:

            candidates = entity_candidates

    # ========================================================
    # SPECIFIC TERM FILTER
    # ========================================================

    
    # ========================================================
    # SORT
    # ========================================================

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # ========================================================
    # DOCUMENT RANKING
    # ========================================================

    document_groups = {}

    for item in candidates:

        filename = item["filename"]

        if filename not in document_groups:
            document_groups[filename] = []

        document_groups[filename].append(item)

    document_scores = []

    for filename, items in document_groups.items():

        top_scores = sorted(
            [
                item["score"]
                for item in items
            ],
            reverse=True
        )[:3]

        if not top_scores:
            continue

        # Average of top supporting chunks
        document_score = sum(
            top_scores
        ) / len(top_scores)

        document_scores.append({

            "filename": filename,

            "score": document_score,

            "supporting_chunks": len(items)
        })

    document_scores.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    print("\nDOCUMENT RANKING:")

    for index, doc in enumerate(
        document_scores,
        start=1
    ):

        print(
            f"{index}. "
            f"{doc['filename']} | "
            f"Score: {doc['score']:.4f} | "
            f"Chunks: {doc['supporting_chunks']}"
        )

    # ========================================================
    # DYNAMIC DOCUMENT THRESHOLD
    # ========================================================

    if document_scores:

        top_score = document_scores[0]["score"]

        document_threshold = max(
            RETRIEVAL_THRESHOLD,
            top_score * 0.72
        )

    else:

        document_threshold = RETRIEVAL_THRESHOLD

    # ========================================================
    # FINAL CHUNKS
    # ========================================================

    final_chunks = []

    per_source = {}

    for item in candidates:

        filename = item["filename"]

        doc_score = next(
            (
                d["score"]
                for d in document_scores
                if d["filename"] == filename
            ),
            0.0
        )

        if doc_score < document_threshold:
            continue

        if item["score"] < RETRIEVAL_THRESHOLD:
            continue

        count = per_source.get(
            filename,
            0
        )

        if count >= MAX_PER_SOURCE:
            continue

        final_chunks.append(item)

        per_source[filename] = count + 1

        if len(final_chunks) >= TOP_K:
            break

    # ========================================================
    # RELATED DOCUMENTS
    # ========================================================

    related_documents = []

    for doc in document_scores:

        if doc["score"] >= document_threshold:

            related_documents.append({
                "filename": doc["filename"],
                "score": round(
                    doc["score"],
                    4
                ),
                "supporting_chunks":
                    doc["supporting_chunks"]
            })

    # ========================================================
    # ALL DOCUMENTS
    # ========================================================

    all_documents = get_uploaded_documents()

    related_names = {
        doc["filename"]
        for doc in related_documents
    }

    unrelated_documents = []

    for doc in all_documents:

        if doc["filename"] not in related_names:

            unrelated_documents.append({
                "filename": doc["filename"],
                "message":
                    "No relevant chunks retrieved",
                "best_score": 0.0
            })

    print("\nFINAL CHUNKS:")

    for index, item in enumerate(
        final_chunks,
        start=1
    ):

        print(
            f"{index}. "
            f"{item['filename']} | "
            f"Page {item['page']} | "
            f"Score {item['score']:.4f}"
        )

    return {

        "chunks": final_chunks,

        "related_documents":
            related_documents,

        "unrelated_documents":
            unrelated_documents
    }


# ============================================================
# LLM
# ============================================================

def generate_answer(
    question: str,
    chunks: list
):

    if not chunks:

        return (
            "I could not find sufficiently relevant "
            "information in the uploaded research documents "
            "to answer this question."
        )

    context_parts = []

    for index, item in enumerate(
        chunks,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {index}
Document: {item['filename']}
Page: {item['page']}

{item['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    prompt = f"""
You are an AI research assistant.

Answer the user's question ONLY using the
research context provided below.

Do not invent information.

If the context does not contain enough
information, explicitly say so.

Cite supporting sources using:
[Source 1], [Source 2], etc.

User question:
{question}

Research context:
{context}

Give a concise but useful research answer.
"""

    hf_token = os.environ.get(
        "HF_TOKEN"
    )

    if not hf_token:

        return build_fallback_answer(
            chunks
        )

    try:

        from openai import OpenAI

        client = OpenAI(
            base_url=
                "https://router.huggingface.co/v1",
            api_key=hf_token
        )

        response = client.chat.completions.create(

            model=
                "openai/gpt-oss-120b:groq",

            messages=[
                {
                    "role":
                        "system",
                    "content":
                        "You are a research assistant. "
                        "Use only supplied sources."
                },
                {
                    "role":
                        "user",
                    "content":
                        prompt
                }
            ],

            max_tokens=700
        )

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        if answer:
            return answer

    except Exception as e:

        print(
            "LLM unavailable:",
            e
        )

    return build_fallback_answer(
        chunks
    )


# ============================================================
# FALLBACK ANSWER
# ============================================================

def build_fallback_answer(chunks):

    answer_parts = []

    answer_parts.append(
        "The retrieved research context contains "
        "the following relevant information:"
    )

    for index, item in enumerate(
        chunks[:3],
        start=1
    ):

        text = item["text"]

        if len(text) > 700:
            text = text[:700] + "..."

        answer_parts.append(
            f"\n[Source {index}] "
            f"{item['filename']} — "
            f"Page {item['page']}\n"
            f"{text}"
        )

    return "\n".join(
        answer_parts
    )


# ============================================================
# CITATION COUNT
# ============================================================

def count_citations(answer: str):

    if not answer:
        return 0

    matches = re.findall(
        r"\[Source\s+\d+\]",
        answer
    )

    return len(
        set(matches)
    )


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "name":
            "AI Research Assistant",

        "version":
            "2.0",

        "status":
            "online",

        "documents":
            collection.count()
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status":
            "healthy",

        "documents":
            collection.count()
    }


# ============================================================
# DOCUMENT LIST
# ============================================================

@app.get("/documents")
def documents():

    return {
        "documents":
            get_uploaded_documents(),

        "count":
            len(get_uploaded_documents())
    }


# ============================================================
# UPLOAD MULTIPLE PDFs
# ============================================================

@app.post("/upload")
async def upload_pdfs(
    files: List[UploadFile] = File(...)
):

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No PDF files selected."
        )

    results = []

    for uploaded_file in files:

        if not uploaded_file.filename:
            continue

        if not uploaded_file.filename.lower().endswith(
            ".pdf"
        ):
            continue

        filename = safe_filename(
            uploaded_file.filename
        )

        destination = (
            UPLOAD_DIR / filename
        )

        try:

            with open(
                destination,
                "wb"
            ) as buffer:

                shutil.copyfileobj(
                    uploaded_file.file,
                    buffer
                )

            # ----------------------------------------
            # Remove old chunks for same filename
            # ----------------------------------------

            try:

                existing = collection.get(
                    where={
                        "filename":
                            filename
                    },
                    include=[]
                )

                old_ids = existing.get(
                    "ids",
                    []
                )

                if old_ids:
                    collection.delete(
                        ids=old_ids
                    )

            except Exception:
                pass

            # ----------------------------------------
            # Index PDF
            # ----------------------------------------

            result = index_pdf(
                destination
            )

            results.append({

                "filename":
                    filename,

                "status":
                    "Indexed",

                "chunks_added":
                    result["chunks_added"]
            })

        except Exception as e:

            print(
                "Upload error:",
                e
            )

            if destination.exists():

                try:
                    destination.unlink()
                except Exception:
                    pass

            results.append({

                "filename":
                    filename,

                "status":
                    "Failed",

                "error":
                    str(e)
            })

    return {

        "message":
            "PDF upload and indexing completed.",

        "uploaded":
            results,

        "total_documents":
            len(get_uploaded_documents()),

        "total_chunks":
            collection.count()
    }


# ============================================================
# REFRESH / CLEAR RESEARCH SET
# ============================================================

@app.post("/refresh")
def refresh_research_set():

    global collection

    print(
        "\n=========================================="
    )
    print(
        "CLEARING RESEARCH SET"
    )
    print(
        "=========================================="
    )

    # --------------------------------------------
    # Delete Chroma collection
    # --------------------------------------------

    try:

        chroma_client.delete_collection(
            name=COLLECTION_NAME
        )

    except Exception as e:

        print(
            "Collection delete warning:",
            e
        )

    # --------------------------------------------
    # Recreate collection
    # --------------------------------------------

    collection = (
        chroma_client
        .get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine"
            }
        )
    )

    # --------------------------------------------
    # Delete uploaded PDFs
    # --------------------------------------------

    deleted_files = 0

    for pdf in UPLOAD_DIR.glob(
        "*.pdf"
    ):

        try:

            pdf.unlink()

            deleted_files += 1

        except Exception as e:

            print(
                "Could not delete:",
                pdf,
                e
            )

    print(
        f"Deleted PDFs: {deleted_files}"
    )

    return {

        "message":
            "Research set cleared successfully.",

        "deleted_documents":
            deleted_files,

        "documents_remaining":
            0,

        "chunks_remaining":
            collection.count()
    }


# ============================================================
# ASK QUESTION
# ============================================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    if collection.count() == 0:

        return {

            "answer":
                "Please upload at least one "
                "research PDF before asking a question.",

            "retrieved_chunks": 0,

            "unique_sources": 0,

            "avg_similarity": 0.0,

            "citations_used": 0,

            "related_documents": [],

            "unrelated_documents": [],

            "sources": []
        }

    retrieval = retrieve_documents(
        question
    )

    chunks = retrieval["chunks"]

    answer = generate_answer(
        question,
        chunks
    )

    # --------------------------------------------
    # Metrics
    # --------------------------------------------

    unique_sources = len(
        set(
            item["filename"]
            for item in chunks
        )
    )

    avg_similarity = (

        sum(
            item["semantic_similarity"]
            for item in chunks
        ) / len(chunks)

        if chunks

        else 0.0
    )

    sources = []

    for index, item in enumerate(
        chunks,
        start=1
    ):

        sources.append({

            "source":
                index,

            "filename":
                item["filename"],

            "page":
                item["page"],

            "similarity":
                round(
                    item["semantic_similarity"],
                    4
                ),

            "score":
                round(
                    item["score"],
                    4
                )
        })

    return {

        "answer":
            answer,

        "retrieved_chunks":
            len(chunks),

        "unique_sources":
            unique_sources,

        "avg_similarity":
            round(
                avg_similarity,
                4
            ),

        "citations_used":
            count_citations(answer),

        "related_documents":
            retrieval[
                "related_documents"
            ],

        "unrelated_documents":
            retrieval[
                "unrelated_documents"
            ],

        "sources":
            sources
    }


# ============================================================
# UI
# ============================================================

if STATIC_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(
            directory=str(STATIC_DIR)
        ),
        name="static"
    )


@app.get("/ui")
def ui():

    index_file = (
        STATIC_DIR / "index.html"
    )

    if not index_file.exists():

        raise HTTPException(
            status_code=404,
            detail="index.html not found."
        )

    return FileResponse(
        index_file
    )
# ============================================================
# DELETE SINGLE PDF
# ============================================================

@app.delete("/documents/{filename}")
def delete_document(filename: str):

    filename = os.path.basename(filename)

    print("=" * 50)
    print("DELETING SINGLE PDF")
    print("=" * 50)
    print("Filename:", filename)

    try:
        collection.delete(
            where={
                "filename": filename
            }
        )

        pdf_path = os.path.join(
            UPLOAD_DIR,
            filename
        )

        if os.path.exists(pdf_path):
            os.remove(pdf_path)

        print("Deleted:", filename)

        return {
            "success": True,
            "message": f"{filename} removed successfully."
        }

    except Exception as e:

        print("DELETE ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    