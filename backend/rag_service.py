"""
MindSaathi AI Service - RAG Educational Layer with Gemini
========================================================
Uses Google Gemini (gemini-1.5-flash) for intelligent Q&A.
Falls back to static knowledge base if Gemini is unavailable.

Guardrails:
  - Refuses diagnosis requests
  - Refuses medication/treatment requests
  - References trusted health organization sources only
"""

import os
import logging
from typing import Optional

from knowledge_base.guardrails import check_guardrails
from knowledge_base.index import retrieve_relevant_chunks

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

SYSTEM_PROMPT = """You are NeuroBot, an educational assistant for MindSaathi — a cognitive wellness screening tool.

STRICT RULES (never violate):
1. You do NOT diagnose diseases. Never say "You have [disease]".
2. You do NOT provide medical treatment advice or medication recommendations.
3. You explain cognitive performance indicators in simple, reassuring, non-alarming language.
4. You always recommend consulting a neurologist for clinical decisions.
5. Reference only: NIH, WHO, Alzheimer's Association, Parkinson's Foundation.
6. If a user seems distressed, show empathy and recommend professional help.
7. Keep responses concise (2-4 sentences simple, up to 6 for complex questions).
8. Be warm, supportive, and educational.

SAFE LANGUAGE:
- Instead of "You have Alzheimer's" → "Some memory performance indicators were noted"
- Instead of "You need medication" → "A neurologist can discuss appropriate next steps"
- Instead of "This confirms dementia" → "These results suggest further professional evaluation may be helpful"

MindSaathi measures 5 cognitive domains:
- Speech Analysis: WPM, pause ratio, speech variability, start delay
- Memory Test: recall accuracy, delayed recall, latency, order matching
- Reaction Time: processing speed, consistency, miss rate
- Stroop Test: executive function, cognitive flexibility, inhibitory control
- Motor Tap Test: rhythmic motor consistency, hand coordination
"""

DISCLAIMER = "\n\n⚠️ *This is NOT medical advice. Always consult a qualified neurologist for clinical evaluation.*"


def _try_gemini(question: str, context: str, user_context: Optional[dict]) -> Optional[str]:
    """Try to get a response from Gemini. Returns None if unavailable."""
    if not GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=SYSTEM_PROMPT,
        )

        ctx_str = ""
        if user_context:
            parts = []
            if user_context.get("user_name"):
                parts.append(f"User: {user_context['user_name']}")
            if user_context.get("recent_scores"):
                parts.append(f"Recent scores: {user_context['recent_scores']}")
            if parts:
                ctx_str = f"\n[User context: {', '.join(parts)}]"

        knowledge_str = f"\n\n[Knowledge base]:\n{context}" if context else ""
        full_prompt = f"Question: {question}{ctx_str}{knowledge_str}"

        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=400,
                temperature=0.3,
            ),
        )
        return response.text.strip() if response.text else None
    except Exception as e:
        logger.warning(f"Gemini API call failed: {e}")
        return None


def _static_answer(question: str, chunks: list, user_context: Optional[dict]) -> str:
    """Compose a safe educational answer from retrieved chunks (fallback)."""
    intro = "Based on trusted neurological health sources:\n\n"
    body = "\n\n".join(c["text"] for c in chunks[:2])
    outro = (
        "\n\nFor personalized evaluation, please consult a neurologist "
        "or visit alz.org (Alzheimer's Association) or parkinson.org (Parkinson's Foundation)."
    )
    return intro + body + outro


def answer_educational_question(
    question: str,
    user_context: Optional[dict] = None,
) -> dict:
    """
    Answer a user's educational question about cognitive health.
    Uses Gemini if available, falls back to static knowledge base.
    """
    logger.info(f"RAG query: {question[:80]}")

    # ── Guardrail check ────────────────────────────────────────────────────
    guardrail_result = check_guardrails(question)
    if guardrail_result["blocked"]:
        logger.warning(f"Guardrail triggered: {guardrail_result['reason']}")
        return {
            "answer": guardrail_result["safe_response"] + DISCLAIMER,
            "sources": [],
            "guardrail_triggered": True,
            "reason": guardrail_result["reason"],
            "disclaimer": DISCLAIMER,
        }

    # ── Retrieve relevant chunks ───────────────────────────────────────────
    chunks = retrieve_relevant_chunks(question, top_k=3)
    context = "\n\n".join(f"[{c['source']}]: {c['text']}" for c in chunks) if chunks else ""
    sources = [c["source"] for c in chunks]

    # ── Try Gemini first, fall back to static ─────────────────────────────
    gemini_answer = _try_gemini(question, context, user_context)

    if gemini_answer:
        answer = gemini_answer
    elif chunks:
        answer = _static_answer(question, chunks, user_context)
    else:
        answer = (
            "I don't have specific information about that in my knowledge base. "
            "I recommend the Alzheimer's Association (alz.org), "
            "NIH National Institute on Aging (nia.nih.gov), or a neurologist."
        )

    return {
        "answer": answer + DISCLAIMER,
        "sources": sources,
        "guardrail_triggered": False,
        "disclaimer": DISCLAIMER,
        "powered_by": "gemini" if gemini_answer else "static",
    }


