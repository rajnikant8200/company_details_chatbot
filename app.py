import json
import math
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# New imports for embeddings and FAISS
import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # handled at runtime

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
ROOT = Path(__file__).parent
KNOWLEDGE_PATH = ROOT / "company_knowledge.txt"
INDEX_PATH = ROOT / "faiss.index"

# --- Helpers ---
STOP_WORDS = set([
    "the","a","an","and","or","of","for","to","in","on","at","by",
    "is","are","what","which","who","how","do","does","your","you","we",
])

def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Embeddings and FAISS ---
EMBEDDING_MODEL = "models/text-embedding-004"  # Correct Gemini embeddings ID


def embed_texts(texts: List[str], *, task_type: str = "retrieval_document") -> np.ndarray:
    if not GEMINI_API_KEY:
        raise RuntimeError("gemini_not_configured")
    vectors: List[List[float]] = []
    for t in texts:
        resp = genai.embed_content(model=EMBEDDING_MODEL, content=t, task_type=task_type)
        vec = resp["embedding"] if isinstance(resp, dict) else getattr(resp, "embedding", None)
        if vec is None:
            raise RuntimeError("empty_embedding_response")
        vectors.append(vec)
    return np.array(vectors, dtype="float32")


def chunk_text(text: str, max_tokens: int = 180) -> List[str]:
    words = normalize_text(text).split(" ")
    chunks: List[str] = []
    buf: List[str] = []
    for w in words:
        buf.append(w)
        if len(buf) >= max_tokens:
            chunks.append(" ".join(buf).strip())
            buf = []
    if buf:
        chunks.append(" ".join(buf).strip())
    return [c for c in chunks if c]

# Global in-memory store
DOC_CHUNKS: List[str] = []
FAISS_INDEX = None
LAST_SIZE = 0


def build_or_load_index() -> Tuple[Any, List[str]]:
    global FAISS_INDEX, DOC_CHUNKS, LAST_SIZE
    text = KNOWLEDGE_PATH.read_text(encoding="utf-8") if KNOWLEDGE_PATH.exists() else ""
    LAST_SIZE = len(text)
    chunks = chunk_text(text, max_tokens=180)
    if not chunks:
        FAISS_INDEX = None
        DOC_CHUNKS = []
        return None, []

    # Embed document chunks
    embeddings = embed_texts(chunks, task_type="retrieval_document")

    if faiss is None:
        raise RuntimeError("faiss_not_installed")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    normed = embeddings / norms
    index.add(normed)

    FAISS_INDEX = index
    DOC_CHUNKS = chunks
    return FAISS_INDEX, DOC_CHUNKS


def ensure_index_loaded():
    global FAISS_INDEX, LAST_SIZE
    if not KNOWLEDGE_PATH.exists():
        FAISS_INDEX = None
        DOC_CHUNKS.clear()
        return
    current_size = KNOWLEDGE_PATH.stat().st_size
    if FAISS_INDEX is None or current_size != LAST_SIZE:
        build_or_load_index()


def expand_query(q: str) -> List[str]:
    qn = normalize_text(q)
    expansions = [qn]
    if "name" in qn:
        expansions.append("company name brand name harivarsh import & export pvt. ltd.")
    if "where" in qn or "location" in qn or "address" in qn:
        expansions.append("headquarters city address ahmedabad gujarat india")
    if "contact" in qn or "email" in qn or "phone" in qn:
        expansions.append("contact details email info@harivarshexport.com phone +91-8200197199")
    if "product" in qn or "service" in qn or "export" in qn:
        expansions.append("offerings grains seeds pulses fertilizers farm equipment")
    if "shipping" in qn or "delivery" in qn or "days" in qn:
        expansions.append("dispatch within seven working days 7 days timing")
    if "payment" in qn or "pay" in qn:
        expansions.append("payment methods bank transfer upi credit cards")
    if "established" in qn or "founded" in qn or "year" in qn:
        expansions.append("established in 2022 founded 2022")
    return expansions


def search_similar(question: str, k: int = 10, min_sim: float = 0.0) -> List[Dict[str, Any]]:
    ensure_index_loaded()
    if FAISS_INDEX is None or not DOC_CHUNKS:
        return []
    queries = expand_query(question)
    collected: Dict[int, float] = {}
    for q in queries:
        q_vec = embed_texts([q], task_type="retrieval_query")
        q_norm = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12)
        D, I = FAISS_INDEX.search(q_norm, k)
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1 or score < min_sim:
                continue
            collected[idx] = max(collected.get(idx, 0.0), float(score))
    # sort by best score desc
    ranked = sorted(collected.items(), key=lambda x: x[1], reverse=True)[:k]
    return [{"id": idx, "score": sc, "text": DOC_CHUNKS[idx]} for idx, sc in ranked]


def best_fallback_context(lines: int = 2) -> str:
    if not KNOWLEDGE_PATH.exists():
        return ""
    try:
        content = KNOWLEDGE_PATH.read_text(encoding="utf-8")
        return "\n".join(content.strip().splitlines()[:lines])
    except Exception:
        return ""


# Simple rule-based extractor for FAQs
NAME_RE = re.compile(r"harivarsh import \& export pvt\. ltd\.", re.I)
CITY_RE = re.compile(r"ahmedabad\b.*gujarat", re.I)
EMAIL_RE = re.compile(r"info@harivarshexport\.com", re.I)
PHONE_RE = re.compile(r"\+91-?82001-?97199", re.I)
EST_RE = re.compile(r"founded in\s+2022|established in\s+2022", re.I)
SHIPPING_RE = re.compile(r"seven working days|7 working days", re.I)
PAYMENT_RE = re.compile(r"bank transfer|upi|credit cards", re.I)
OFFERINGS_RE = re.compile(r"grains|seeds|pulses|fertilizers|farm equipment", re.I)


def extract_rule_based(question: str, knowledge: str) -> str | None:
    q = question.lower()
    if any(k in q for k in ["name", "company name", "what is the company name"]):
        if NAME_RE.search(knowledge):
            return "Our company name is Harivarsh Import & Export Pvt. Ltd."
    if any(k in q for k in ["where", "location", "address"]):
        if CITY_RE.search(knowledge):
            return "We are headquartered in Ahmedabad, Gujarat, India."
    if any(k in q for k in ["contact", "email", "phone"]):
        parts = []
        if EMAIL_RE.search(knowledge):
            parts.append("Email: info@harivarshexport.com")
        if PHONE_RE.search(knowledge):
            parts.append("Phone: +91-82001-97199")
        if parts:
            return "; ".join(parts)
    if any(k in q for k in ["established", "founded", "year"]):
        if EST_RE.search(knowledge):
            return "Harivarsh Import & Export was established in 2022."
    if any(k in q for k in ["shipping", "delivery"]):
        if SHIPPING_RE.search(knowledge):
            return "Typical dispatch is within approximately 7 working days after order confirmation."
    if any(k in q for k in ["payment", "pay", "methods"]):
        if PAYMENT_RE.search(knowledge):
            return "We accept bank transfer, UPI, and major credit cards."
    if any(k in q for k in ["products", "offer", "export", "what do you export", "what we export"]):
        if OFFERINGS_RE.search(knowledge):
            return "We deal in agricultural products including grains, seeds, pulses, fertilizers, and farm equipment."
    return None


class AskBody(BaseModel):
    question: str


@app.post("/ask")
async def ask(body: AskBody):
    question = body.question
    try:
        if not GEMINI_API_KEY:
            return {"answer": "I don't know based on the provided information.", "meta": {"error": "gemini_not_configured"}}

        neighbors = search_similar(question, k=10)
        if neighbors:
            context = "\n".join([f"[ctx#{i+1} score={n['score']:.3f}] {n['text']}" for i, n in enumerate(neighbors)])
        else:
            raw = KNOWLEDGE_PATH.read_text(encoding="utf-8") if KNOWLEDGE_PATH.exists() else ""
            rb = extract_rule_based(question, raw)
            if rb:
                return {"answer": rb, "meta": {"provider": "rules", "rag": False}}
            fallback_ctx = best_fallback_context(lines=3)
            if not fallback_ctx:
                return {"answer": "I don't know based on the provided information.", "meta": {"rag": False, "neighbors": []}}
            context = fallback_ctx

        instruction = (
            "Answer as Harivarsh Import & Export assistant."
            "\n- Be concise and precise."
            "\n- Use only the CONTEXT; do not invent details."
            "\n- If the CONTEXT does not contain the answer, reply: I don't know based on the provided information."
            "\n- Where relevant, include exact figures, timeframes, or contact details from the CONTEXT."
        )
        prompt = f"{instruction}\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nFINAL ANSWER:"

        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content(prompt)
        text = getattr(getattr(result, "response", None), "text", lambda: "")()
        if not text:
            return {"answer": "I don't know based on the provided information.", "meta": {"rag": True, "neighbors": neighbors}}

        # Append light citations from top 2 contexts
        cites = " | ".join([f"ctx#{i+1}:{n['score']:.2f}" for i, n in enumerate(neighbors[:2])]) if neighbors else ""
        if cites:
            text = f"{text}\n\nSources: {cites}"
        return {"answer": text, "meta": {"provider": "gemini", "rag": True, "neighbors": neighbors}}
    except Exception as e:
        try:
            raw = KNOWLEDGE_PATH.read_text(encoding="utf-8") if KNOWLEDGE_PATH.exists() else ""
            rb = extract_rule_based(question, raw)
            if rb:
                return {"answer": rb, "meta": {"provider": "rules", "error": str(e)}}
        except Exception:
            pass
        return {"answer": "I don't know based on the provided information.", "meta": {"rag": False, "error": str(e)}}


@app.post("/reindex")
async def reindex():
    try:
        build_or_load_index()
        return {"status": "ok", "chunks": len(DOC_CHUNKS)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/")
def root():
    return {"status": "Harivarsh Python RAG+FAISS (txt-only) server is running!"} 