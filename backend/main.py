# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging

# Suppress verbose output from RAG chain
logging.getLogger('langchain').setLevel(logging.ERROR)

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
    allow_origins=["*"],   # for development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema

class QuestionRequest(BaseModel):
    question: str

# Health check

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
