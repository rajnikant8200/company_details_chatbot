import os
import google.generativeai as genai
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from dotenv import load_dotenv  # Imports the function to read .env files

# --- Configuration ---
# This line loads the variables from your .env file (e.g., GOOGLE_API_KEY)
load_dotenv()

# The script now reads the key loaded from the .env file
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("CRITICAL ERROR: GOOGLE_API_KEY not found.")
    print("Please make sure you have a .env file in your main project folder with your key in it.")
    exit()

# Initialize the Generative Model
model = genai.GenerativeModel('gemini-1.5-flash')

# --- FastAPI App Setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Company Knowledge ---
ROOT = Path(__file__).parent
KNOWLEDGE_PATH = ROOT.parent / "company_knowledge.txt"

def load_knowledge() -> str:
    """Reads the knowledge file and returns its content."""
    if KNOWLEDGE_PATH.exists():
        return KNOWLEDGE_PATH.read_text(encoding="utf-8")
    return "Error: The company_knowledge.txt file was not found. Please ensure it is in the correct directory."

# --- API Endpoints ---
class AskBody(BaseModel):
    question: str

@app.post("/ask")
async def ask(body: AskBody):
    """
    Handles a user's question by sending it to the Gemini API
    with the company's knowledge as context.
    """
    question = body.question.strip()
    if not question:
        return {"answer": "Please ask a question."}

    company_context = load_knowledge()

    prompt = f"""
    You are a professional, helpful chatbot for a company called "Harivarsh Import & Export Pvt. Ltd.".
    Your role is to answer user questions based ONLY on the information provided below.
    If the answer is not in the information, you must clearly state that you do not have that information.
    Do not make up answers. Keep your responses concise and friendly.

    --- COMPANY INFORMATION ---
    {company_context}
    --- END OF INFORMATION ---

    USER QUESTION: "{question}"

    ANSWER:
    """

    try:
        response = model.generate_content(prompt)
        return {"answer": response.text}
    except Exception as e:
        print(f"An error occurred while calling the Gemini API: {e}")
        return {"answer": "Sorry, I'm having trouble connecting to the AI service right now. Please try again later."}

@app.get("/")
async def read_index():
    """Serves the main HTML file."""
    html_path = ROOT.parent / "index.html"
    if not html_path.exists():
        return {"error": "index.html not found"}
    return FileResponse(html_path)

