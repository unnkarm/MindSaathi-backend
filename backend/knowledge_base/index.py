"""
MindSaathi RAG - Knowledge Base Index
=====================================
NEW FILE v2.0

Provides retrieve_relevant_chunks() for the RAG pipeline.

In production:
  - Replace this with a vector database (Pinecone, Chroma, FAISS)
  - Embed documents from: NIH, WHO, Alzheimer's Association, Parkinson's Foundation
  - Use sentence-transformers for embedding + cosine similarity retrieval

For now: keyword-based fallback with curated educational snippets.
Only trusted sources are included.
"""

import re
from typing import List

# ── Curated knowledge base ─────────────────────────────────────────────────────
# Source: publicly available educational content from trusted health organizations.
# In production: replace with embedded vector store over full PDF/HTML documents.

KNOWLEDGE_BASE = [
    {
        "id": "mem_001",
        "source": "NIH National Institute on Aging",
        "url": "https://www.nia.nih.gov/health/memory-and-forgetfulness",
        "keywords": ["memory", "recall", "forgetfulness", "delayed recall"],
        "text": (
            "Some degree of memory decline is a normal part of aging. Older adults may take "
            "longer to learn new information and recall names or words. However, significant "
            "memory problems — such as forgetting recent conversations or getting lost in "
            "familiar places — may warrant further evaluation by a healthcare professional."
        ),
    },
    {
        "id": "mem_002",
        "source": "Alzheimer's Association",
        "url": "https://www.alz.org/alzheimers-dementia/10_signs",
        "keywords": ["memory loss", "alzheimer", "warning signs", "recall", "word finding"],
        "text": (
            "One of the most common early signs of Alzheimer's is memory loss that disrupts "
            "daily life, particularly forgetting recently learned information. Difficulty "
            "finding the right words and increased pauses during speech may also be early "
            "indicators worth discussing with a physician."
        ),
    },
    {
        "id": "react_001",
        "source": "NIH — Aging & Reaction Time",
        "url": "https://www.nia.nih.gov",
        "keywords": ["reaction time", "slow", "processing speed", "motor"],
        "text": (
            "Reaction time naturally slows with age due to changes in neural processing speed. "
            "A 70-year-old typically has a reaction time 20–40% slower than a 25-year-old. "
            "This is normal. However, unusually high variability in reaction times — rather "
            "than just slowness — may indicate attention or motor control concerns."
        ),
    },
    {
        "id": "park_001",
        "source": "Parkinson's Foundation",
        "url": "https://www.parkinson.org/understanding-parkinsons/what-is-parkinsons",
        "keywords": ["parkinson", "motor", "tremor", "tapping", "movement", "bradykinesia"],
        "text": (
            "Parkinson's disease primarily affects motor control. Early signs include "
            "bradykinesia (slowed movement), rhythmic tremor at rest, and reduced arm swing. "
            "Cognitive changes may appear later in the disease. A neurologist uses clinical "
            "examination and neuroimaging to diagnose Parkinson's disease."
        ),
    },
    {
        "id": "speech_001",
        "source": "Alzheimer's Association — Communication Changes",
        "url": "https://www.alz.org/alzheimers-dementia/dementia-symptoms",
        "keywords": ["speech", "language", "word finding", "fluency", "pause", "verbal"],
        "text": (
            "Changes in language and communication can be early signs of cognitive decline. "
            "These may include difficulty finding words, increased pauses, reduced verbal "
            "fluency (fewer words generated in a time period), and more frequent use of "
            "filler words. A speech-language pathologist or neurologist can assess these changes."
        ),
    },
    {
        "id": "edu_001",
        "source": "NIH — Cognitive Reserve",
        "url": "https://www.nia.nih.gov/news/cognitive-reserve-and-brain-resilience",
        "keywords": ["education", "cognitive reserve", "resilience", "learning"],
        "text": (
            "Cognitive reserve refers to the brain's ability to compensate for damage. "
            "Higher levels of education, mentally stimulating work, and lifelong learning "
            "are associated with greater cognitive reserve, which can delay the onset of "
            "symptoms even when pathological changes are present in the brain."
        ),
    },
    {
        "id": "lexdiv_001",
        "source": "MindSaathi Educational Content",
        "url": "https://neuroaid.app/education",
        "keywords": ["lexical diversity", "vocabulary", "word variety", "speech score"],
        "text": (
            "Lexical diversity measures the variety of words used in speech. "
            "A higher lexical diversity score indicates a richer vocabulary and more varied "
            "language use, which is associated with stronger language processing. "
            "Declining lexical diversity over time may indicate changes in language centers "
            "of the brain and is one of several indicators assessed in cognitive screening."
        ),
    },
    {
        "id": "fatigue_001",
        "source": "NIH — Fatigue & Cognitive Testing",
        "url": "https://www.nia.nih.gov",
        "keywords": ["fatigue", "sleep", "tired", "retest", "accuracy"],
        "text": (
            "Fatigue, sleep deprivation, and acute illness can significantly affect cognitive "
            "test performance, leading to lower scores that do not reflect baseline ability. "
            "Clinical neuropsychological assessments account for these factors and may "
            "recommend retesting after adequate rest if performance may be compromised."
        ),
    },
]


def retrieve_relevant_chunks(question: str, top_k: int = 3) -> List[dict]:
    """
    Retrieve the most relevant knowledge base chunks for a question.

    In production: replace with vector similarity search.
    Currently: keyword overlap scoring.

    Args:
        question: User's natural language question.
        top_k: Number of chunks to return.

    Returns:
        List of chunk dicts sorted by relevance.
    """
    q_tokens = set(re.findall(r"\b\w+\b", question.lower()))

    scored = []
    for chunk in KNOWLEDGE_BASE:
        kw_set = set(w for kw in chunk["keywords"] for w in kw.split())
        overlap = len(q_tokens & kw_set)
        if overlap > 0:
            scored.append((overlap, chunk))

    scored.sort(key=lambda x: -x[0])
    return [chunk for _, chunk in scored[:top_k]]


