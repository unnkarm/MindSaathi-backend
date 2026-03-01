"""
games.py — NeuroSaathi Games Backend Router
Handles all 9 cognitive games across Memory, Attention, and Problem Solving.

Endpoints:
  GET  /api/games                    - List all games with metadata
  GET  /api/games/{game_id}          - Get a single game definition
  POST /api/games/{game_id}/submit   - Submit a completed game session
  GET  /api/games/history            - User's game history (auth required)
  GET  /api/games/summary            - Aggregated domain scores (auth required)
  GET  /api/games/leaderboard        - Top scores per game (optional auth)
"""

import json
import os
import uuid
import math
import statistics
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, Field

from utils.logger import log_info

router = APIRouter(prefix="/games", tags=["games"])

# ── Data persistence ──────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(os.path.dirname(__file__), "..", "data")
GAME_RESULTS_FILE = os.path.join(DATA_DIR, "game_results.json")
SESSIONS_FILE   = os.path.join(DATA_DIR, "sessions.json")
USERS_FILE      = os.path.join(DATA_DIR, "users.json")
os.makedirs(DATA_DIR, exist_ok=True)


# ── Game catalog (mirrors frontend gamesCatalog.js) ───────────────────────────
GAMES_CATALOG = [
    {
        "id": "game-story-reconstruct",
        "title": "Story Fragment Reconstruction",
        "category": "Memory",
        "icon": "📖",
        "difficulty": "Medium",
        "measures": "Episodic memory, temporal sequencing, context retention",
        "total_questions": 5,
        "time_limit_seconds": 120,
        "scoring": {
            "base_per_correct": 20,
            "time_bonus_threshold_seconds": 60,
            "time_bonus_points": 10,
        },
        "domain": "memory",
        "description": "Reconstruct story fragments in the correct temporal order, testing episodic memory and narrative understanding.",
    },
    {
        "id": "game-face-name",
        "title": "Face-Name Association",
        "category": "Memory",
        "icon": "🧑",
        "difficulty": "Medium",
        "measures": "Associative memory, social memory, encoding accuracy",
        "total_questions": 5,
        "time_limit_seconds": 90,
        "scoring": {
            "base_per_correct": 20,
            "time_bonus_threshold_seconds": 45,
            "time_bonus_points": 10,
        },
        "domain": "memory",
        "description": "Match faces with their associated names, testing associative and social memory encoding.",
    },
    {
        "id": "game-audio-tone",
        "title": "Audio Tone Recall",
        "category": "Memory",
        "icon": "🎵",
        "difficulty": "Hard",
        "measures": "Auditory working memory, sequential recall, processing consistency",
        "total_questions": 5,
        "time_limit_seconds": 90,
        "scoring": {
            "base_per_correct": 20,
            "time_bonus_threshold_seconds": 45,
            "time_bonus_points": 10,
        },
        "domain": "memory",
        "description": "Recall audio tone sequences, measuring auditory working memory and pattern recognition.",
    },
    {
        "id": "game-rhythm-sync",
        "title": "Rhythm Sync Tap",
        "category": "Attention",
        "icon": "🥁",
        "difficulty": "Medium",
        "measures": "Sustained attention, motor-attention coordination, reaction stability",
        "total_questions": 5,
        "time_limit_seconds": 120,
        "scoring": {
            "base_per_correct": 20,
            "time_bonus_threshold_seconds": 60,
            "time_bonus_points": 10,
        },
        "domain": "attention",
        "description": "Tap in sync with a rhythm to test sustained attention and motor-attention coordination.",
    },
    {
        "id": "game-countdown-interrupt",
        "title": "Countdown Interrupt",
        "category": "Attention",
        "icon": "⏱️",
        "difficulty": "Hard",
        "measures": "Vigilance, alertness, response inhibition",
        "total_questions": 5,
        "time_limit_seconds": 90,
        "scoring": {
            "base_per_correct": 20,
            "time_bonus_threshold_seconds": 45,
            "time_bonus_points": 10,
        },
        "domain": "attention",
        "description": "Detect and respond to signals during a countdown, testing vigilance and response inhibition.",
    },
    {
        "id": "game-multi-rule-sort",
        "title": "Multi-Rule Sorting",
        "category": "Attention",
        "icon": "🧩",
        "difficulty": "Hard",
        "measures": "Selective attention, rule maintenance, switching accuracy",
        "total_questions": 5,
        "time_limit_seconds": 120,
        "scoring": {
            "base_per_correct": 20,
            "time_bonus_threshold_seconds": 60,
            "time_bonus_points": 10,
        },
        "domain": "attention",
        "description": "Sort items by shifting rules, measuring selective attention and cognitive flexibility.",
    },
    {
        "id": "game-gk-logic",
        "title": "Timed GK Logic Challenge",
        "category": "Problem Solving",
        "icon": "🌍",
        "difficulty": "Medium",
        "measures": "Semantic memory, retrieval speed, logical elimination, processing time",
        "total_questions": 5,
        "time_limit_seconds": 90,
        "scoring": {
            "base_per_correct": 20,
            "time_bonus_threshold_seconds": 45,
            "time_bonus_points": 10,
        },
        "domain": "problem_solving",
        "description": "Answer general knowledge questions under time pressure, measuring semantic memory and retrieval speed.",
    },
    {
        "id": "game-calc-sprint",
        "title": "Mental Calculation Sprint",
        "category": "Problem Solving",
        "icon": "🔢",
        "difficulty": "Hard",
        "measures": "Working memory, numerical reasoning, processing speed, executive function",
        "total_questions": 5,
        "time_limit_seconds": 90,
        "scoring": {
            "base_per_correct": 20,
            "time_bonus_threshold_seconds": 40,
            "time_bonus_points": 10,
        },
        "domain": "problem_solving",
        "description": "Solve mental arithmetic problems rapidly, testing working memory and numerical reasoning.",
    },
    {
        "id": "game-decision-sim",
        "title": "Real-Life Decision Simulation",
        "category": "Problem Solving",
        "icon": "🧠",
        "difficulty": "Medium",
        "measures": "Applied numeracy, planning, practical reasoning, judgment",
        "total_questions": 5,
        "time_limit_seconds": 120,
        "scoring": {
            "base_per_correct": 20,
            "time_bonus_threshold_seconds": 60,
            "time_bonus_points": 10,
        },
        "domain": "problem_solving",
        "description": "Solve realistic daily-life decision scenarios, testing practical reasoning and judgment.",
    },
]

GAME_MAP: Dict[str, dict] = {g["id"]: g for g in GAMES_CATALOG}

CATEGORY_DOMAINS = {
    "Memory": "memory",
    "Attention": "attention",
    "Problem Solving": "problem_solving",
}

DIFFICULTY_MULTIPLIERS = {
    "Easy":   1.0,
    "Medium": 1.2,
    "Hard":   1.5,
}


# ── Schemas ───────────────────────────────────────────────────────────────────

class AnswerDetail(BaseModel):
    question_index: int
    selected_option: int
    correct_option: int
    time_taken_ms: Optional[float] = None  # ms per question


class GameSubmitRequest(BaseModel):
    answers: List[AnswerDetail]
    total_time_seconds: float = Field(..., ge=0)
    session_metadata: Optional[Dict[str, Any]] = None  # e.g. device, browser


class QuestionResult(BaseModel):
    question_index: int
    correct: bool
    time_taken_ms: Optional[float]
    selected_option: int
    correct_option: int


class GameSessionResult(BaseModel):
    session_id: str
    game_id: str
    game_title: str
    category: str
    domain: str
    difficulty: str
    correct_count: int
    total_questions: int
    accuracy_pct: float
    base_score: float
    time_bonus: float
    difficulty_multiplier: float
    final_score: float          # 0–100
    total_time_seconds: float
    avg_time_per_question_ms: Optional[float]
    reaction_speed_score: float  # derived from avg response time
    consistency_score: float     # derived from std dev of response times
    cognitive_load_index: float  # composite of accuracy + speed + consistency
    question_results: List[QuestionResult]
    completed_at: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        try:
            return json.load(f)
        except Exception:
            return {}


def _save(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _user_from_token(token: str) -> Optional[dict]:
    sessions = _load(SESSIONS_FILE)
    session  = sessions.get(token)
    if not session:
        return None
    users = _load(USERS_FILE)
    return users.get(session["user_id"])


def _compute_game_score(
    answers: List[AnswerDetail],
    game_meta: dict,
    total_time_seconds: float,
) -> GameSessionResult:
    """
    Score a game session:
      - base_score    = correct / total × 100
      - time_bonus    = extra points if completed under threshold
      - final_score   = (base_score + time_bonus) × difficulty_multiplier, capped at 100
      - reaction_speed_score  = derived from mean response time
      - consistency_score     = derived from std dev of response times
      - cognitive_load_index  = weighted combo of accuracy + speed + consistency
    """
    scoring      = game_meta["scoring"]
    diff_mult    = DIFFICULTY_MULTIPLIERS.get(game_meta["difficulty"], 1.0)
    total_q      = game_meta["total_questions"]

    question_results = []
    correct_count = 0
    times_ms: List[float] = []

    for a in answers:
        is_correct = (a.selected_option == a.correct_option)
        if is_correct:
            correct_count += 1
        if a.time_taken_ms is not None:
            times_ms.append(a.time_taken_ms)
        question_results.append(QuestionResult(
            question_index=a.question_index,
            correct=is_correct,
            time_taken_ms=a.time_taken_ms,
            selected_option=a.selected_option,
            correct_option=a.correct_option,
        ))

    accuracy_pct = round((correct_count / total_q) * 100, 1)
    base_score   = round(accuracy_pct, 1)

    # Time bonus
    time_bonus = 0.0
    if total_time_seconds <= scoring["time_bonus_threshold_seconds"]:
        # Scale bonus: full bonus at 0s, zero at threshold
        ratio      = 1.0 - (total_time_seconds / scoring["time_bonus_threshold_seconds"])
        time_bonus = round(scoring["time_bonus_points"] * ratio, 1)

    # Final score capped at 100
    final_score = round(min(100.0, (base_score + time_bonus) * diff_mult), 1)

    # Reaction speed score (0–100): faster = higher
    avg_time_ms = None
    reaction_speed_score = 50.0
    if times_ms:
        avg_time_ms = round(statistics.mean(times_ms), 1)
        # Reference: 500ms = great, 3000ms = poor (for MCQ)
        reaction_speed_score = round(max(0.0, min(100.0, 100 - ((avg_time_ms - 500) / 25))), 1)

    # Consistency score (0–100): lower std dev = more consistent = higher score
    consistency_score = 50.0
    if len(times_ms) >= 2:
        std_ms = statistics.stdev(times_ms)
        # < 200ms std = perfect, > 2000ms std = poor
        consistency_score = round(max(0.0, min(100.0, 100 - (std_ms / 20))), 1)

    # Cognitive load index: weighted
    cognitive_load_index = round(
        0.50 * accuracy_pct
        + 0.25 * reaction_speed_score
        + 0.25 * consistency_score,
        1,
    )

    return GameSessionResult(
        session_id=str(uuid.uuid4()),
        game_id=game_meta["id"],
        game_title=game_meta["title"],
        category=game_meta["category"],
        domain=game_meta["domain"],
        difficulty=game_meta["difficulty"],
        correct_count=correct_count,
        total_questions=total_q,
        accuracy_pct=accuracy_pct,
        base_score=base_score,
        time_bonus=time_bonus,
        difficulty_multiplier=diff_mult,
        final_score=final_score,
        total_time_seconds=round(total_time_seconds, 2),
        avg_time_per_question_ms=avg_time_ms,
        reaction_speed_score=reaction_speed_score,
        consistency_score=consistency_score,
        cognitive_load_index=cognitive_load_index,
        question_results=question_results,
        completed_at=datetime.utcnow().isoformat(),
    )


def _compute_domain_summary(sessions: List[dict]) -> Dict[str, Any]:
    """
    Aggregate game sessions into per-domain cognitive scores and trends.
    Returns a dict suitable for frontend dashboard consumption.
    """
    domain_scores: Dict[str, List[float]] = {
        "memory": [], "attention": [], "problem_solving": [],
    }
    category_scores: Dict[str, List[float]] = {}

    for s in sessions:
        d = s.get("domain")
        fs = s.get("final_score")
        if d and fs is not None:
            domain_scores.setdefault(d, []).append(fs)
        cat = s.get("category")
        if cat and fs is not None:
            category_scores.setdefault(cat, []).append(fs)

    def _agg(scores: List[float]) -> Dict[str, Any]:
        if not scores:
            return {"avg": None, "best": None, "sessions": 0, "trend": "no_data"}
        avg  = round(statistics.mean(scores), 1)
        best = round(max(scores), 1)
        # Trend: compare last 3 vs previous 3
        trend = "stable"
        if len(scores) >= 6:
            recent = statistics.mean(scores[-3:])
            prior  = statistics.mean(scores[-6:-3])
            if recent > prior + 5:
                trend = "improving"
            elif recent < prior - 5:
                trend = "declining"
        return {"avg": avg, "best": best, "sessions": len(scores), "trend": trend}

    # Overall cognitive score (equal weighting across domains)
    all_scores = [s for ss in domain_scores.values() for s in ss]
    overall_avg = round(statistics.mean(all_scores), 1) if all_scores else None

    return {
        "overall_cognitive_score": overall_avg,
        "total_sessions": len(sessions),
        "domains": {
            "memory":          _agg(domain_scores["memory"]),
            "attention":       _agg(domain_scores["attention"]),
            "problem_solving": _agg(domain_scores["problem_solving"]),
        },
        "by_category": {cat: _agg(sc) for cat, sc in category_scores.items()},
        "recent_sessions": sessions[-5:][::-1],  # last 5, newest first
    }


def _compute_leaderboard(all_results: dict) -> List[dict]:
    """Compute top scores per game across all users."""
    # game_id -> list of {user_id, score, completed_at}
    game_tops: Dict[str, List[dict]] = {}

    # Load users for display names
    users = _load(USERS_FILE)

    for uid, sessions in all_results.items():
        user_name = users.get(uid, {}).get("full_name", "Anonymous")
        for s in sessions:
            gid   = s.get("game_id")
            score = s.get("final_score")
            if gid and score is not None:
                game_tops.setdefault(gid, []).append({
                    "user_id":      uid,
                    "display_name": user_name,
                    "score":        score,
                    "completed_at": s.get("completed_at"),
                })

    leaderboard = {}
    for gid, entries in game_tops.items():
        top10 = sorted(entries, key=lambda e: e["score"], reverse=True)[:10]
        leaderboard[gid] = top10

    return leaderboard


def _compute_cognitive_domain_scores(sessions: List[dict]) -> Dict[str, float]:
    """
    Returns domain scores in the same 0-100 range as the analyze endpoint,
    so they can be merged/compared with clinical assessment data.
    """
    domain_avgs = {}
    for domain in ["memory", "attention", "problem_solving"]:
        domain_sessions = [s for s in sessions if s.get("domain") == domain]
        if domain_sessions:
            scores = [s["cognitive_load_index"] for s in domain_sessions if s.get("cognitive_load_index") is not None]
            domain_avgs[domain] = round(statistics.mean(scores), 1) if scores else None
        else:
            domain_avgs[domain] = None
    return domain_avgs


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("")
def list_games(
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
):
    """
    List all available games.
    Optional filters: ?category=Memory&difficulty=Hard
    """
    games = GAMES_CATALOG
    if category:
        games = [g for g in games if g["category"].lower() == category.lower()]
    if difficulty:
        games = [g for g in games if g["difficulty"].lower() == difficulty.lower()]
    return {
        "games": games,
        "total": len(games),
        "categories": list({g["category"] for g in GAMES_CATALOG}),
        "difficulties": list({g["difficulty"] for g in GAMES_CATALOG}),
    }


@router.get("/summary")
def get_game_summary(authorization: str = Header(...)):
    """
    Get the authenticated user's aggregated game performance summary,
    broken down by domain, category, and trend.
    """
    token = authorization.replace("Bearer ", "").strip()
    user  = _user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized. Please log in.")

    all_results = _load(GAME_RESULTS_FILE)
    sessions    = all_results.get(user["id"], [])

    summary = _compute_domain_summary(sessions)
    domain_scores = _compute_cognitive_domain_scores(sessions)

    return {
        "user_id": user["id"],
        "summary": summary,
        "cognitive_domain_scores": domain_scores,
        "games_played": list({s["game_id"] for s in sessions}),
        "total_play_time_seconds": round(
            sum(s.get("total_time_seconds", 0) for s in sessions), 1
        ),
    }


@router.get("/history")
def get_game_history(
    authorization: str = Header(...),
    game_id: Optional[str] = None,
    limit: int = 20,
):
    """
    Get the authenticated user's game session history.
    Optional filter: ?game_id=game-story-reconstruct&limit=10
    """
    token = authorization.replace("Bearer ", "").strip()
    user  = _user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized. Please log in.")

    all_results = _load(GAME_RESULTS_FILE)
    sessions    = all_results.get(user["id"], [])

    if game_id:
        sessions = [s for s in sessions if s.get("game_id") == game_id]

    sessions_sorted = sessions[-limit:][::-1]  # newest first

    return {
        "sessions": sessions_sorted,
        "total": len(sessions),
        "filtered_total": len(sessions_sorted),
    }


@router.get("/leaderboard")
def get_leaderboard(
    game_id: Optional[str] = None,
    authorization: Optional[str] = Header(default=None),
):
    """
    Get top scores per game or for a specific game.
    Optional: ?game_id=game-calc-sprint
    """
    all_results = _load(GAME_RESULTS_FILE)
    leaderboard = _compute_leaderboard(all_results)

    if game_id:
        if game_id not in GAME_MAP:
            raise HTTPException(status_code=404, detail=f"Game '{game_id}' not found.")
        return {
            "game_id":   game_id,
            "game_title": GAME_MAP[game_id]["title"],
            "top_scores": leaderboard.get(game_id, []),
        }

    return {"leaderboard": leaderboard}


@router.get("/{game_id}")
def get_game(game_id: str):
    """Get a single game definition with metadata and questions count."""
    game = GAME_MAP.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail=f"Game '{game_id}' not found.")
    return {"game": game}


@router.post("/{game_id}/submit", response_model=GameSessionResult)
def submit_game(
    game_id: str,
    payload: GameSubmitRequest,
    authorization: Optional[str] = Header(default=None),
):
    """
    Submit a completed game session and receive a scored result.

    Accepts:
      - answers: list of {question_index, selected_option, correct_option, time_taken_ms}
      - total_time_seconds: wall-clock time for the whole game
      - session_metadata: optional device/context info

    Returns:
      - Full GameSessionResult with accuracy, scores, cognitive metrics.
    """
    game = GAME_MAP.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail=f"Game '{game_id}' not found.")

    if len(payload.answers) == 0:
        raise HTTPException(status_code=422, detail="No answers provided.")

    if len(payload.answers) > game["total_questions"]:
        raise HTTPException(
            status_code=422,
            detail=f"Too many answers. Game has {game['total_questions']} questions.",
        )

    log_info(f"[/api/games/{game_id}/submit] scoring {len(payload.answers)} answers")

    result = _compute_game_score(payload.answers, game, payload.total_time_seconds)

    # Persist result if user is authenticated
    if authorization:
        token = authorization.replace("Bearer ", "").strip()
        user  = _user_from_token(token)
        if user:
            all_results = _load(GAME_RESULTS_FILE)
            uid         = user["id"]
            history     = all_results.get(uid, [])
            record      = result.model_dump()
            record["session_metadata"] = payload.session_metadata
            history.append(record)
            all_results[uid] = history[-100:]  # keep last 100 sessions
            _save(GAME_RESULTS_FILE, all_results)
            log_info(f"[games] saved session {result.session_id} for user {uid}")

    return result


@router.get("/{game_id}/my-stats")
def get_my_game_stats(
    game_id: str,
    authorization: str = Header(...),
):
    """
    Get the authenticated user's personal stats for a specific game:
    best score, average score, number of attempts, improvement trend.
    """
    game = GAME_MAP.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail=f"Game '{game_id}' not found.")

    token = authorization.replace("Bearer ", "").strip()
    user  = _user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized. Please log in.")

    all_results = _load(GAME_RESULTS_FILE)
    sessions    = [
        s for s in all_results.get(user["id"], [])
        if s.get("game_id") == game_id
    ]

    if not sessions:
        return {
            "game_id":    game_id,
            "game_title": game["title"],
            "attempts":   0,
            "best_score": None,
            "avg_score":  None,
            "trend":      "no_data",
            "sessions":   [],
        }

    scores = [s["final_score"] for s in sessions]
    trend  = "stable"
    if len(scores) >= 4:
        recent = statistics.mean(scores[-2:])
        prior  = statistics.mean(scores[-4:-2])
        if recent > prior + 5:
            trend = "improving"
        elif recent < prior - 5:
            trend = "declining"

    return {
        "game_id":       game_id,
        "game_title":    game["title"],
        "category":      game["category"],
        "domain":        game["domain"],
        "attempts":      len(sessions),
        "best_score":    round(max(scores), 1),
        "avg_score":     round(statistics.mean(scores), 1),
        "latest_score":  round(scores[-1], 1),
        "trend":         trend,
        "accuracy_avg":  round(statistics.mean([s["accuracy_pct"] for s in sessions]), 1),
        "avg_time_secs": round(statistics.mean([s["total_time_seconds"] for s in sessions]), 1),
        "sessions":      sessions[-10:][::-1],  # last 10, newest first
    }


@router.get("/{game_id}/leaderboard")
def get_game_leaderboard(game_id: str):
    """Get the top 10 scores for a specific game."""
    if game_id not in GAME_MAP:
        raise HTTPException(status_code=404, detail=f"Game '{game_id}' not found.")

    all_results = _load(GAME_RESULTS_FILE)
    leaderboard = _compute_leaderboard(all_results)

    return {
        "game_id":    game_id,
        "game_title": GAME_MAP[game_id]["title"],
        "top_scores": leaderboard.get(game_id, []),
    }