import json
import math
import os
import re
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# No external API usage. Answers are generated locally from the knowledge file.

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

def split_into_sentences(raw_text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", raw_text.strip())
    sentences = [p.strip() for p in parts if p and p.strip()]
    return sentences

def load_sentences() -> List[str]:
    text = KNOWLEDGE_PATH.read_text(encoding="utf-8") if KNOWLEDGE_PATH.exists() else ""
    return split_into_sentences(text)


def select_sentences(question: str, context: str, max_sentences: int = 3) -> str:
    q_terms = [t for t in normalize_text(question).split(" ") if t and t not in STOP_WORDS]
    if not q_terms:
        return context.strip()
    sentences = re.split(r"(?<=[.!?])\s+", context)
    scored: List[tuple[float, str]] = []
    for s in sentences:
        s_clean = normalize_text(s)
        if not s_clean:
            continue
        overlap = sum(1 for t in q_terms if t in s_clean)
        length_penalty = max(len(s_clean.split(" ")), 1) ** 0.25
        score = overlap / length_penalty
        if overlap > 0:
            scored.append((score, s.strip()))
    if not scored:
        return context.strip()
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [s for _, s in scored[:max_sentences]]
    return " \n".join(chosen)


def expand_query(q: str) -> List[str]:
    qn = normalize_text(q)
    # Common typos and variants
    qn = (
        qn.replace("expot", "export")
           .replace("exprot", "export")
           .replace("expart", "export")
           .replace("cntry", "country")
           .replace("compny", "company")
           .replace("adress", "address")
           .replace("which country to you export", "which countries do you export to")
           .replace("what we export", "what do you export")
           .replace("what you provide", "what products do you provide")
    )
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
    if "expand" in qn or "market" in qn or "reach" in qn:
        expansions.append("market reach expansion growth demand quality expectations")
    if "document" in qn or "amend" in qn or "order" in qn:
        expansions.append("documentation amendments orders written confirmation scope terms")
    if "supplier" in qn or "relationship" in qn or "partnership" in qn:
        expansions.append("supplier relationships collaboration sustainable practices")
    if "acknowledge" in qn or "inbound" in qn or "query" in qn:
        expansions.append("inbound queries acknowledgement business hours target products")
    if "exchange" in qn or "rate" in qn or "currency" in qn:
        expansions.append("exchange rate considerations quotation phase settlement")
    if "quality" in qn or "standard" in qn or "certification" in qn:
        expansions.append("quality management export grade benchmarks traceability")
    if "country" in qn or "destination" in qn or "market" in qn:
        expansions.append("countries asia europe africa export destinations")
    if "logistics" in qn or "dispatch" in qn or "timeline" in qn:
        expansions.append("logistics planning dispatch working days route selection")
    if "compliance" in qn or "regulation" in qn:
        expansions.append("compliance regulations documentation export standards")
    if "mission" in qn or "value" in qn or "purpose" in qn:
        expansions.append("mission values reliability responsiveness transparency")
    
    # Country-specific expansions
    if any(k in qn for k in ["bangladesh", "bangladesh"]):
        expansions.append("bangladesh nepal sri lanka grains food products")
    if any(k in qn for k in ["germany", "netherlands", "europe"]):
        expansions.append("germany netherlands europe farm equipment renewable")
    if any(k in qn for k in ["africa", "kenya", "tanzania", "ghana"]):
        expansions.append("africa kenya tanzania ghana fertilizers soil")
    if any(k in qn for k in ["vietnam", "philippines", "indonesia"]):
        expansions.append("vietnam philippines indonesia seeds cultivation")
    if any(k in qn for k in ["uae", "saudi", "oman"]):
        expansions.append("uae saudi arabia oman pulses grades")
    
    # Product-specific expansions
    if any(k in qn for k in ["grains", "wheat", "rice", "maize"]):
        expansions.append("grains wheat rice maize bangladesh nepal sri lanka")
    if any(k in qn for k in ["seeds", "cultivation"]):
        expansions.append("seeds cultivation vietnam philippines indonesia")
    if any(k in qn for k in ["pulses", "protein"]):
        expansions.append("pulses protein uae saudi oman grades")
    if any(k in qn for k in ["fertilizers", "soil"]):
        expansions.append("fertilizers soil kenya tanzania ghana")
    if any(k in qn for k in ["equipment", "machinery"]):
        expansions.append("equipment machinery germany netherlands farm")
    
    # Time-related expansions
    if any(k in qn for k in ["time", "open", "hours", "when"]):
        expansions.append("business hours phone support contact time")
    if any(k in qn for k in ["started", "begin", "opened", "established"]):
        expansions.append("established founded 2022 started began")
    
    return expansions


def search_similar(question: str, k: int = 12, min_sim: float = 0.0) -> List[Dict[str, Any]]:
    sentences = load_sentences()
    if not sentences:
        return []
    # Prepare docs
    doc_tokens: List[List[str]] = [normalize_text(c).split(" ") for c in sentences]
    q_tokens = normalize_text(question).split(" ")
    # Score by token overlap (simple RAG)
    results: List[Dict[str, Any]] = []
    doc_sets = [set(t for t in toks if t and t not in STOP_WORDS) for toks in doc_tokens]
    q_set = [t for t in q_tokens if t and t not in STOP_WORDS]
    for idx, dset in enumerate(doc_sets):
        if not dset:
            continue
        overlap = sum(1 for t in q_set if t in dset)
        score = overlap / (len(q_set) ** 0.5 * max(len(dset), 1) ** 0.5) if q_set else 0.0
        if score >= min_sim:
            results.append({"id": idx, "score": float(score), "text": sentences[idx]})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:k]


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
EST_RE = re.compile(r"founded in\s+2022|established in\s+2022|started in\s+2022|started\s+2022", re.I)
SHIPPING_RE = re.compile(r"seven working days|7 working days", re.I)
PAYMENT_RE = re.compile(r"bank transfer|upi|credit cards", re.I)
OFFERINGS_RE = re.compile(r"grains|seeds|pulses|fertilizers|farm equipment", re.I)


def extract_rule_based(question: str, knowledge: str) -> str | None:
    q = question.lower()
    q = (
        q.replace("expot", "export")
         .replace("exprot", "export")
         .replace("expart", "export")
         .replace("compny", "company")
         .replace("adress", "address")
         .replace("cntry", "country")
    )
    # Quick intents
    if any(k in q for k in ["what we export", "what you export", "what do you export", "what you provide", "what products", "what do you provide", "products", "provide"]):
        # Try to pull a concise sentence from knowledge
        m = re.search(r"(deal[s]? in .*?\.)", knowledge, re.I)
        if m:
            return m.group(1).strip()
        return "We deal in agricultural products including grains, seeds, pulses, fertilizers, and farm equipment."
    if any(k in q for k in ["which country", "which countries", "country to export", "countries do you export", "export to which"]):
        m = re.search(r"(serve[s]? over .*?\.|export destinations.*?\.)", knowledge, re.I)
        if m:
            return m.group(1).strip()
        return "We export across Asia, Europe, and Africa, including South and Southeast Asian markets."
    q = q.replace("expot", "export").replace("compny", "company").replace("adress", "address")
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
    if any(k in q for k in ["established", "founded", "year", "started"]):
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
    
    # Enhanced rule-based answers for complex questions
    if any(k in q for k in ["expand", "market reach", "growth", "expansion"]):
        return "The company continues to expand market reach in line with demand, aligning offerings with export-grade quality expectations and treating client communication as a critical component of service."
    
    if any(k in q for k in ["document", "amendments", "orders", "changes"]):
        return "Amendments to orders are documented via revised confirmations. The firm prioritizes clear written confirmation of scope and terms."
    
    if any(k in q for k in ["supplier", "relationships", "long-term", "partnership"]):
        return "Supplier relationships are cultivated for long-term collaboration. The firm encourages sustainable practices within its supply network."
    
    if any(k in q for k in ["acknowledge", "inbound", "queries", "customer", "process"]):
        return "Inbound queries receive acknowledgement within standard business hours. The team requests target products and estimated quantities in inquiries."
    
    if any(k in q for k in ["exchange", "rate", "currency", "considerations"]):
        return "Exchange rate considerations are addressed in the quotation phase. Settlement timelines are communicated within commercial offers."
    
    if any(k in q for k in ["quality", "standards", "certification"]):
        return "Quality management emphasizes export-grade benchmarks and traceability. All agricultural products meet export standards and are quality-certified."
    
    if any(k in q for k in ["countries", "export", "destinations", "markets"]):
        return "The company serves over fifteen countries across Asia, Europe, and Africa. It also supplies select buyers in Europe with export-compliant goods."
    
    if any(k in q for k in ["logistics", "shipping", "dispatch", "timeline"]):
        return "Logistics planning aims to dispatch orders within seven working days after confirmation. Route selection is optimized for cost and transit reliability."
    
    if any(k in q for k in ["compliance", "regulations", "documentation"]):
        return "Harivarsh supports international documentation for export compliance. The firm aligns with applicable Indian export regulations."
    
    if any(k in q for k in ["mission", "values", "purpose"]):
        return "The company's mission is to connect agricultural supply with global demand. Its values emphasize reliability, responsiveness, and transparency."
    
    # Country-specific export questions
    if any(k in q for k in ["bangladesh", "bangladesh"]):
        return "The company serves markets in Bangladesh, Nepal, and Sri Lanka for grains and food products."
    if any(k in q for k in ["germany", "netherlands", "europe"]):
        return "Farm equipment and renewable items are delivered to Germany and Netherlands. European markets receive export-compliant goods under strict quality standards."
    if any(k in q for k in ["africa", "african", "kenya", "tanzania", "ghana"]):
        return "Fertilizers and soil-improvement inputs are supplied to Kenya, Tanzania, and Ghana. African markets are a significant growth region for agricultural exports."
    if any(k in q for k in ["vietnam", "philippines", "indonesia", "southeast"]):
        return "Certified cultivation seeds are shipped to Vietnam, Philippines, and Indonesia."
    if any(k in q for k in ["uae", "saudi", "oman", "middle east"]):
        return "Pulses in multiple grades are exported to United Arab Emirates, Saudi Arabia, and Oman."
    if any(k in q for k in ["asia", "asian", "countries"]):
        return "The company serves over fifteen countries across Asia, Europe, and Africa. Major export destinations include South Asian and Southeast Asian countries."
    
    # Product-specific questions
    if any(k in q for k in ["grains", "wheat", "rice", "maize"]):
        return "Core agricultural products include grains procured from vetted suppliers. The company serves markets in Bangladesh, Nepal, and Sri Lanka for grains and food products."
    if any(k in q for k in ["seeds", "cultivation"]):
        return "Seeds for cultivation and wholesale distribution are part of the catalog. Certified cultivation seeds are shipped to Vietnam, Philippines, and Indonesia."
    if any(k in q for k in ["pulses", "protein"]):
        return "Pulses are offered in several grades to match buyer specifications. Pulses in multiple grades are exported to United Arab Emirates, Saudi Arabia, and Oman."
    if any(k in q for k in ["fertilizers", "soil"]):
        return "Fertilizers are supplied with documentation aligned to export norms. Fertilizers and soil-improvement inputs are supplied to Kenya, Tanzania, and Ghana."
    if any(k in q for k in ["equipment", "farm equipment", "machinery"]):
        return "Farm equipment is available subject to availability and demand cycles. Farm equipment and renewable items are delivered to Germany and Netherlands."
    
    # Time-related questions
    if any(k in q for k in ["time", "open", "hours", "business", "when"]):
        return "Inbound queries receive acknowledgement within standard business hours. Phone support is available at +91-82001-97199 during business hours."
    if any(k in q for k in ["started", "begin", "opened", "launched"]):
        return "Harivarsh Import & Export was established in 2022."
    
    return None


class AskBody(BaseModel):
    question: str


@app.post("/ask")
async def ask(body: AskBody):
    question = body.question
    try:
        # Try rule-based first for direct answers
        raw = KNOWLEDGE_PATH.read_text(encoding="utf-8") if KNOWLEDGE_PATH.exists() else ""
        rb = extract_rule_based(question, raw)
        if rb:
            return {"answer": rb, "meta": {"provider": "rules", "rag": False}}
        
        # If no rule-based match, try retrieval (top sentences only)
        neighbors = search_similar(question, k=5)
        if neighbors:
            context = " ".join([n['text'] for n in neighbors])
        else:
            fallback_ctx = best_fallback_context(lines=3)
            if not fallback_ctx:
                return {"answer": "I don't know based on the provided information.", "meta": {"rag": False, "neighbors": []}}
            context = fallback_ctx
        answer = select_sentences(question, context, max_sentences=3)
        if not answer.strip():
            return {"answer": "I don't know based on the provided information.", "meta": {"rag": True, "neighbors": neighbors}}
        return {"answer": answer, "meta": {"provider": "local", "rag": True, "neighbors": neighbors}}
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
    # No indexing step anymore; kept for compatibility
    try:
        sentences = load_sentences()
        return {"status": "ok", "chunks": len(sentences)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/")
def root():
    return {"status": "Harivarsh Python local RAG (txt-only, no API) is running!"}