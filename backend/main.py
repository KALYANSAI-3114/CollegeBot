# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import logging
import os

# Suppress verbose output from RAG chain
logging.getLogger("langchain").setLevel(logging.ERROR)

from rag_pipeline import answer_question

# Initialize FastAPI app
app = FastAPI(
    title="VVIT RAG Chatbot API",
    description="FastAPI backend for VVIT Information Chatbot using RAG",
    version="1.0.0"
)

# Enable CORS (for HTML frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # OK for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- FRONTEND SERVING (Render compatible) --------

FRONTEND_DIR = os.path.abspath("../frontend")

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static"
)

@app.get("/ui")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# -------- API SCHEMA --------

class QuestionRequest(BaseModel):
    question: str

# Health check (Render uses this)
@app.get("/")
def health_check():
    return {"status": "VVIT RAG API is running"}

# Main RAG endpoint
@app.post("/ask")
def ask_question(data: QuestionRequest):
    """
    Receives a question from frontend,
    sends it to the RAG pipeline,
    returns the answer.
    """
    result = answer_question(data.question)
    return result
