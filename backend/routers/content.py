
import json, os, uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, Header

router = APIRouter(tags=["content"])

DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
CONTENT_FILE = os.path.join(DATA_DIR, "custom_content.json")
SESSIONS_FILE= os.path.join(DATA_DIR, "sessions.json")
USERS_FILE   = os.path.join(DATA_DIR, "users.json")
os.makedirs(DATA_DIR, exist_ok=True)

def _load(path):
    if not os.path.exists(path): return {}
    with open(path) as f:
        try: return json.load(f)
        except: return {}

def _load_content():
    if not os.path.exists(CONTENT_FILE):
        return {"passages": [], "word_sets": []}
    with open(CONTENT_FILE) as f:
        try: return json.load(f)
        except: return {"passages": [], "word_sets": []}

def _save_content(c):
    with open(CONTENT_FILE, "w") as f: json.dump(c, f, indent=2)

def _auth(authorization: str):
    token = authorization.replace("Bearer ", "").strip()
    sessions = _load(SESSIONS_FILE)
    session  = sessions.get(token)
    if not session: raise HTTPException(status_code=401, detail="Unauthorized.")
    users = _load(USERS_FILE)
    user = users.get(session["user_id"])
    if not user: raise HTTPException(status_code=401, detail="Unauthorized.")
    return user


# ── GET /api/content  — get all custom content (any logged-in user) ─────────
@router.get("/content")
def get_content(authorization: str = Header(...)):
    _auth(authorization)
    return _load_content()


# ── POST /api/content/passage  — doctor adds a custom passage ───────────────
@router.post("/content/passage")
def add_passage(body: dict, authorization: str = Header(...)):
    user = _auth(authorization)
    if user.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can add content.")

    text = (body.get("text") or "").strip()
    if len(text) < 30:
        raise HTTPException(status_code=400, detail="Passage must be at least 30 characters.")
    if len(text) > 800:
        raise HTTPException(status_code=400, detail="Passage must be under 800 characters.")

    c = _load_content()
    c.setdefault("passages", []).append({
        "id":         str(uuid.uuid4()),
        "text":       text,
        "added_by":   user["full_name"],
        "added_role": user.get("role"),
        "created_at": datetime.utcnow().isoformat(),
    })
    _save_content(c)
    return {"ok": True, "count": len(c["passages"])}


# ── POST /api/content/wordset  — doctor adds a custom word set ──────────────
@router.post("/content/wordset")
def add_wordset(body: dict, authorization: str = Header(...)):
    user = _auth(authorization)
    if user.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can add content.")

    words = body.get("words", [])
    if not isinstance(words, list) or len(words) < 8:
        raise HTTPException(status_code=400, detail="Provide at least 8 words.")
    if len(words) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 words per set.")

    words = [str(w).strip().capitalize() for w in words if str(w).strip()]

    c = _load_content()
    c.setdefault("word_sets", []).append({
        "id":         str(uuid.uuid4()),
        "words":      words,
        "added_by":   user["full_name"],
        "added_role": user.get("role"),
        "created_at": datetime.utcnow().isoformat(),
    })
    _save_content(c)
    return {"ok": True, "count": len(c["word_sets"])}


# ── DELETE /api/content/passage/{id} ────────────────────────────────────────
@router.delete("/content/passage/{item_id}")
def delete_passage(item_id: str, authorization: str = Header(...)):
    user = _auth(authorization)
    if user.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can delete content.")
    c = _load_content()
    c["passages"] = [p for p in c.get("passages", []) if p["id"] != item_id]
    _save_content(c)
    return {"ok": True}


# ── DELETE /api/content/wordset/{id} ────────────────────────────────────────
@router.delete("/content/wordset/{item_id}")
def delete_wordset(item_id: str, authorization: str = Header(...)):
    user = _auth(authorization)
    if user.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can delete content.")
    c = _load_content()
    c["word_sets"] = [w for w in c.get("word_sets", []) if w["id"] != item_id]
    _save_content(c)
    return {"ok": True}