"""
rag_engine.py — Health Information RAG Engine
Uses google-genai SDK (Python 3.14 compatible) + FAISS for Gemini-powered RAG.
"""

import os
import pickle
import re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

import numpy as np
import faiss
from google import genai
from google.genai import types

load_dotenv()

# ─── Configuration ────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
KB_DIR = Path(__file__).parent / "knowledge_base"
INDEX_DIR = Path(__file__).parent / "health_vectordb"
INDEX_FILE = INDEX_DIR / "faiss.index"
DOCS_FILE = INDEX_DIR / "documents.pkl"

EMBED_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.0-flash"
EMBED_BATCH = 50
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

# Initialize client
client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

# ─── Critical Condition Keywords ──────────────────────────────────────────────
CRITICAL_KEYWORDS = [
    "chest pain", "chest pressure", "heart attack", "difficulty breathing",
    "can't breathe", "cannot breathe", "shortness of breath", "stroke",
    "face drooping", "arm weakness", "speech difficulty", "slurred speech",
    "thunderclap headache", "worst headache", "uncontrolled bleeding",
    "vomiting blood", "blood in stool", "blood in vomit", "coughing blood",
    "loss of consciousness", "collapsed", "seizure", "convulsion",
    "severe allergic reaction", "throat swelling", "anaphylaxis",
    "severe abdominal pain", "appendicitis", "high fever stiff neck",
    "fever stiff neck", "suicidal", "self-harm", "overdose", "poisoning",
    "blue lips", "lips turning blue", "not breathing", "pregnancy bleeding",
    "fetal movement stopped", "infant fever", "baby fever", "child breathing",
    "child seizure", "meningitis", "sepsis", "blood poisoning",
]

# ─── Out-of-Domain Keywords ───────────────────────────────────────────────────
NON_HEALTH_KEYWORDS = [
    "stock", "crypto", "cryptocurrency", "bitcoin", "forex",
    "food recipe", "travel", "hotel", "flight", "booking",
    "movie", "music", "game", "sport score", "weather",
    "election", "politics", "code", "programming", "debug",
    "javascript", "python tutorial", "math", "algebra",
    "history", "geography", "law", "legal advice",
]


# ─── Text Chunking ─────────────────────────────────────────────────────────────
def _chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ─── Knowledge Base Loading ───────────────────────────────────────────────────
def load_knowledge_base() -> list[str]:
    if not KB_DIR.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {KB_DIR}")
    txt_files = list(KB_DIR.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in: {KB_DIR}")
    all_chunks = []
    for path in txt_files:
        text = path.read_text(encoding="utf-8")
        all_chunks.extend(_chunk_text(text))
    return all_chunks


# ─── Gemini Embeddings (new SDK) ─────────────────────────────────────────────
def _embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed document chunks using gemini-embedding-001."""
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        all_embeddings.extend([e.values for e in response.embeddings])
    return all_embeddings


def _embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values


# ─── FAISS Vector Store ───────────────────────────────────────────────────────
def build_vector_store(force_rebuild: bool = False):
    """Build or load FAISS index with Gemini embeddings. Returns (index, documents)."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_FILE.exists() and DOCS_FILE.exists() and not force_rebuild:
        index = faiss.read_index(str(INDEX_FILE))
        with open(DOCS_FILE, "rb") as f:
            documents = pickle.load(f)
        if index.ntotal > 0:
            return index, documents

    documents = load_knowledge_base()
    embeddings = _embed_documents(documents)

    vectors = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(INDEX_FILE))
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(documents, f)

    return index, documents


def retrieve_context(query: str, store: tuple, k: int = 5) -> str:
    """Retrieve top-k relevant chunks via cosine similarity."""
    index, documents = store
    q_vec = np.array([_embed_query(query)], dtype=np.float32)
    faiss.normalize_L2(q_vec)
    _, indices = index.search(q_vec, k)
    results = [documents[i] for i in indices[0] if 0 <= i < len(documents)]
    return "\n\n---\n\n".join(results)


# ─── Domain and Safety Gates ──────────────────────────────────────────────────
def is_critical_condition(query: str) -> bool:
    q_lower = query.lower()
    return any(kw in q_lower for kw in CRITICAL_KEYWORDS)


def is_health_query(query: str) -> bool:
    q_lower = query.lower()
    for kw in NON_HEALTH_KEYWORDS:
        if kw in q_lower:
            return False
    if not client:
        return True
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=(
                "You are a strict domain classifier. Is the following query related to "
                "HEALTH, MEDICAL SYMPTOMS, MEDICINES, or GENERAL WELLNESS? "
                f"Query: {query}\n"
                "Reply ONLY 'YES' or 'NO'."
            ),
        )
        return response.text.strip().upper().startswith("YES")
    except Exception:
        return True


# ─── Response Generation ──────────────────────────────────────────────────────
def generate_response(
    query: str,
    context: str,
    age_group: str = "Adult",
    severity: str = "Mild",
    duration: str = "Not specified",
    image_analysis: Optional[str] = None,
) -> dict:
    if not client:
        return _error_response("API key not configured. Add GOOGLE_API_KEY to .env file.")
    if not context.strip():
        return _error_response(
            "There is insufficient verified medical information to provide a reliable answer."
        )

    image_section = f"\n\nMEDICINE IMAGE ANALYSIS:\n{image_analysis}" if image_analysis else ""

    system_prompt = """You are a Health Information RAG Assistant under STRICT medical safety rules.

ABSOLUTE RULES:
1. NEVER diagnose. Use only: "These symptoms may be associated with..."
2. NEVER prescribe medicines or suggest dosages.
3. NEVER invent symptoms, conditions, drugs, or treatments.
4. ALL responses grounded ONLY in the provided Retrieved Medical Evidence.
5. If confidence < 70%, default to doctor referral.
6. Suggest ONLY: rest, hydration, balanced food, monitoring, avoiding triggers, seeking medical advice.
7. Always include expiry/safety reminders when medicine is discussed.

RESPONSE FORMAT — Use EXACTLY these 6 sections:
**1. Query Understanding**
**2. Retrieved Medical Evidence Summary**
**3. Symptom / Image Observation**
**4. Possible Associations (Non-Diagnostic)**
**5. Safe Guidance**
**6. Confidence Level**"""

    user_prompt = (
        f"Retrieved Medical Evidence:\n{context}{image_section}\n\n"
        f"User Query: {query}\n"
        f"Patient Context: Age Group = {age_group}, Severity = {severity}, Duration = {duration}\n\n"
        "Generate a structured health information response. No diagnosis, no prescriptions."
    )

    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        raw_text = response.text.strip()
        return {"success": True, "raw": raw_text, "sections": _parse_sections(raw_text)}
    except Exception as e:
        return _error_response(f"Response generation failed: {str(e)}")


def _parse_sections(text: str) -> dict:
    sections = {
        "query_understanding": "", "evidence_summary": "", "observation": "",
        "possible_associations": "", "safe_guidance": "", "confidence": "",
    }
    keys = list(sections.keys())
    headers = [
        "1. Query Understanding", "2. Retrieved Medical Evidence Summary",
        "3. Symptom / Image Observation", "4. Possible Associations (Non-Diagnostic)",
        "5. Safe Guidance", "6. Confidence Level",
    ]
    for i, header in enumerate(headers):
        start_idx = text.find(f"**{header}**")
        if start_idx == -1:
            start_idx = text.find(f"**{i+1}.")
            if start_idx == -1:
                continue
        next_start = len(text)
        for j in range(i + 1, len(headers)):
            ns = text.find(f"**{headers[j]}**")
            if ns == -1:
                ns = text.find(f"**{j+1}.")
            if ns != -1:
                next_start = ns
                break
        content = "\n".join(text[start_idx:next_start].split("\n")[1:]).strip()
        sections[keys[i]] = content
    return sections


def _error_response(message: str) -> dict:
    return {
        "success": False, "raw": message,
        "sections": {
            "query_understanding": "", "evidence_summary": "", "observation": "",
            "possible_associations": "", "safe_guidance": message,
            "confidence": "Low — insufficient information to provide a reliable answer.",
        },
    }
