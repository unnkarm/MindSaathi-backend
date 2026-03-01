"""
MindSaathi RAG - Guardrail Rules
================================
NEW FILE v2.0

Blocks or redirects questions that:
  - Ask for a diagnosis ("Do I have Alzheimer's?")
  - Ask for medication advice ("What medicine should I take?")
  - Ask for specific dosage / treatment protocols
  - Request interpretation as a definitive clinical result

If triggered, returns a safe, empathetic redirect response.
"""

import re
from typing import Dict

# Patterns that indicate a diagnosis request
DIAGNOSIS_PATTERNS = [
    r"\bdo i have\b",
    r"\bhave i got\b",
    r"\bam i (suffering|diagnosed)\b",
    r"\bis this (alzheimer|dementia|parkinson)",
    r"\bconfirm(s)? (i have|my)\b",
    r"\bdiagnos",
]

# Patterns that indicate medication / treatment request
MEDICATION_PATTERNS = [
    r"\bwhich (medicine|medication|drug|pill|tablet)\b",
    r"\bwhat (medicine|medication|drug|should i take)\b",
    r"\bshould i take\b",
    r"\bdosage\b",
    r"\bprescri(be|ption)\b",
    r"\btreatment (plan|protocol)\b",
    r"\bcure\b",
]

SAFE_DIAGNOSIS_RESPONSE = (
    "MindSaathi cannot provide a diagnosis. This screening tool identifies "
    "cognitive risk indicators for educational purposes only. "
    "For a clinical diagnosis, please consult a qualified neurologist or physician. "
    "You can find a specialist through your local hospital or the "
    "Alzheimer's Association Helpline: 1-800-272-3900."
)

SAFE_MEDICATION_RESPONSE = (
    "MindSaathi cannot provide medication or treatment advice. "
    "Only a qualified physician or neurologist can recommend appropriate treatment. "
    "Please consult your doctor, who can discuss evidence-based options based on "
    "your complete medical history and clinical evaluation."
)


def check_guardrails(question: str) -> dict:
    """
    Check if a question violates guardrail rules.

    Returns:
        Dict with 'blocked' (bool), 'reason', 'safe_response'.
    """
    q_lower = question.lower()

    for pattern in DIAGNOSIS_PATTERNS:
        if re.search(pattern, q_lower):
            return {
                "blocked": True,
                "reason": "diagnosis_request",
                "safe_response": SAFE_DIAGNOSIS_RESPONSE,
            }

    for pattern in MEDICATION_PATTERNS:
        if re.search(pattern, q_lower):
            return {
                "blocked": True,
                "reason": "medication_request",
                "safe_response": SAFE_MEDICATION_RESPONSE,
            }

    return {"blocked": False, "reason": None, "safe_response": None}



