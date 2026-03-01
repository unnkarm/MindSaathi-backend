"""
MindSaathi Backend - Chat/RAG Router
====================================
POST /api/chat — Educational chatbot backed by the local RAG pipeline.

The RAG logic (guardrails + knowledge base retrieval) now lives directly
in the backend (rag_service.py + knowledge_base/) so no separate
ai-service process is required.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

from rag_service import answer_educational_question
from utils.logger import log_info

router = APIRouter()

DISCLAIMER = (
    "⚠️ This is NOT medical advice. Always consult a qualified neurologist "
    "or physician for clinical evaluation and treatment decisions."
)


class ChatRequest(BaseModel):
    question: str
    user_context: Optional[dict] = None   # age, recent scores, etc.


class ChatResponse(BaseModel):
    answer: str
    sources: list = []
    guardrail_triggered: bool = False
    disclaimer: str = DISCLAIMER


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    """
    Educational chatbot endpoint with RAG retrieval.

    Guardrails:
    - Refuses diagnosis requests
    - Refuses medication/treatment requests
    - Only references trusted health organization sources
      (NIH, WHO, Alzheimer's Association, Parkinson's Foundation)
    """
    log_info(f"[/api/chat] question={payload.question[:60]}")

    result = answer_educational_question(payload.question, payload.user_context)

    return ChatResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        guardrail_triggered=result.get("guardrail_triggered", False),
        disclaimer=DISCLAIMER,
    )


