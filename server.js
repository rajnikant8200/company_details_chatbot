import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import fs from "fs";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// Load FAQs once at startup for fallback answers and retrieval
let faqEntries = [];
try {
  const faqsPath = new URL("./faqs.json", import.meta.url);
  const raw = fs.readFileSync(faqsPath, "utf-8");
  faqEntries = JSON.parse(raw);
} catch (e) {
  console.warn("Could not load faqs.json for fallback answers:", e?.message || e);
}

function findFaqAnswer(question) {
  if (!question || !faqEntries?.length) return null;
  const q = String(question).toLowerCase();
  for (const item of faqEntries) {
    const keywords = Array.isArray(item.keywords) ? item.keywords : [];
    for (const kw of keywords) {
      if (q.includes(String(kw).toLowerCase())) {
        return item.answer || null;
      }
    }
  }
  return null;
}

// --- Simple RAG utilities over FAQs ---
function normalizeText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenize(text) {
  const stop = new Set(["the","a","an","and","or","of","for","to","in","on","at","by","is","are","what","which","who","how","do","does","your","you","we"]);
  const tokens = normalizeText(text).split(" ");
  return tokens.filter(t => t && !stop.has(t));
}

function scoreOverlap(queryTokens, docTokens) {
  if (!queryTokens.length || !docTokens.length) return 0;
  const docSet = new Set(docTokens);
  let overlap = 0;
  for (const t of queryTokens) if (docSet.has(t)) overlap++;
  const score = overlap / Math.sqrt(queryTokens.length * docSet.size);
  return score;
}

// Precompute tokens for FAQs
const faqDocs = (faqEntries || []).map((item, idx) => {
  const content = `${(item.keywords || []).join(" ")} ${item.answer || ""}`;
  return {
    id: idx,
    keywords: item.keywords || [],
    answer: item.answer || "",
    tokens: tokenize(content)
  };
});

function retrieveTopK(question, k = 3) {
  const qTokens = tokenize(question);
  const scored = faqDocs.map(d => ({ d, score: scoreOverlap(qTokens, d.tokens) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
  return scored;
}

function buildContextFromDocs(scoredDocs, minScore = 0.05) {
  const filtered = scoredDocs.filter(s => s.score >= minScore);
  if (!filtered.length) return { context: "", citations: [] };
  const lines = filtered.map((s, i) => `[#${i+1}] Answer: ${faqEntries[s.d.id].answer}`);
  return {
    context: lines.join("\n"),
    citations: filtered.map(s => ({ id: s.d.id, score: Number(s.score.toFixed(3)), answer: faqEntries[s.d.id].answer }))
  };
}

// Provider: Gemini only
const gemini = process.env.GEMINI_API_KEY ? new GoogleGenerativeAI(process.env.GEMINI_API_KEY) : null;

async function askGeminiWithRag(question) {
  if (!gemini) throw new Error("gemini_not_configured");
  const top = retrieveTopK(question, 4);
  const { context, citations } = buildContextFromDocs(top);
  const instruction = "You are an assistant for Harivarsh Import & Export. Answer strictly using the provided CONTEXT. If the CONTEXT does not contain the answer, say: 'I don't know based on the provided information.'";
  const prompt = `${instruction}\n\nCONTEXT:\n${context || "(no relevant context)"}\n\nQUESTION: ${question}\n\nFINAL ANSWER:`;

  const model = gemini.getGenerativeModel({ model: "gemini-1.5-flash" });
  const result = await model.generateContent(prompt);
  const text = result?.response?.text?.();
  if (!text) throw new Error("empty_gemini_response");
  return { answer: text, citations };
}

app.post("/ask", async (req, res) => {
  const { question } = req.body || {};
  try {
    const { answer, citations } = await askGeminiWithRag(question);
    return res.json({ answer, meta: { provider: "gemini", rag: true, citations } });
  } catch (err) {
    console.error(err);
    // Fallback to FAQ if Gemini is unavailable or errors out
    const fallback = findFaqAnswer(question) || "Our AI is currently unavailable. For inquiries, email info@harivarshexport.com or call +91-82001-97199.";
    return res.status(200).json({ answer: fallback, meta: { source: "faq_fallback", rag: false } });
  }
});

app.get("/", (req, res) => {
  res.send("Harivarsh AI chatbot server is running!");
});

app.listen(3000, () => console.log("Server running on http://localhost:3000"));
