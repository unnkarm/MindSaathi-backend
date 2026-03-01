"""
messages.py — MindSaathi Messaging Router
Mounted at /api (no extra prefix), so routes are:
  POST   /api/messages/send
  GET    /api/messages/unread/count   ← MUST be before /{id}
  GET    /api/conversations
  GET    /api/messages/{other_user_id}
  DELETE /api/messages/{message_id}
"""

import json, os, uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Header

router = APIRouter(tags=["messages"])  # NO prefix — mounted under /api directly

DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
MESSAGES_FILE = os.path.join(DATA_DIR, "messages.json")
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")
USERS_FILE    = os.path.join(DATA_DIR, "users.json")
os.makedirs(DATA_DIR, exist_ok=True)

def _load(path):
    if not os.path.exists(path): return {}
    with open(path) as f:
        try: return json.load(f)
        except: return {}

def _load_msgs():
    if not os.path.exists(MESSAGES_FILE): return []
    with open(MESSAGES_FILE) as f:
        try: return json.load(f)
        except: return []

def _save_msgs(msgs):
    with open(MESSAGES_FILE, "w") as f: json.dump(msgs, f, indent=2)

def _user_from_token(token: str):
    sessions = _load(SESSIONS_FILE)
    session  = sessions.get(token)
    if not session: return None
    users = _load(USERS_FILE)
    return users.get(session["user_id"])

def _auth(authorization: str):
    token = authorization.replace("Bearer ", "").strip()
    user  = _user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    return user


# ── POST /api/messages/send ───────────────────────────────────────────────────
@router.post("/messages/send")
def send_message(body: dict, authorization: str = Header(...)):
    user         = _auth(authorization)
    text         = (body.get("text") or "").strip()
    recipient_id = (body.get("recipient_id") or "").strip()

    if not text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    if not recipient_id:
        raise HTTPException(status_code=400, detail="Recipient ID required.")

    # Validate recipient exists
    users = _load(USERS_FILE)
    if recipient_id not in users:
        raise HTTPException(status_code=404, detail=f"Recipient not found.")

    # Enforce enrollment-based messaging access
    sender_data = users.get(user["id"], {})
    if user.get("role") == "doctor":
        enrolled_ids = set(sender_data.get("patient_list", []))
        if recipient_id not in enrolled_ids:
            raise HTTPException(status_code=403, detail="You can only message your enrolled patients.")
    else:
        assigned_doctor = sender_data.get("assigned_doctor_id")
        if recipient_id != assigned_doctor:
            raise HTTPException(status_code=403, detail="You can only message your assigned doctor.")

    msgs = _load_msgs()
    msg  = {
        "id":           str(uuid.uuid4()),
        "sender_id":    user["id"],
        "sender_name":  user["full_name"],
        "sender_role":  user.get("role", "patient"),
        "recipient_id": recipient_id,
        "text":         text,
        "timestamp":    datetime.utcnow().isoformat(),
        # Track read state per-user. Sender has implicitly read it.
        "read_by":      [user["id"]],
        "deleted_by":   [],
    }
    msgs.append(msg)
    _save_msgs(msgs)
    return {"message": msg}


# ── GET /api/messages/unread/count  (STATIC — must be before /{id}) ──────────
@router.get("/messages/unread/count")
def unread_count(authorization: str = Header(...)):
    user  = _auth(authorization)
    uid   = user["id"]
    msgs  = _load_msgs()
    count = sum(1 for m in msgs
                if m["recipient_id"] == uid
                and uid not in m.get("deleted_by", [])
                and uid not in m.get("read_by", []))
    return {"count": count}


# ── GET /api/conversations ────────────────────────────────────────────────────
@router.get("/conversations")
def get_conversations(authorization: str = Header(...)):
    user  = _auth(authorization)
    uid   = user["id"]
    msgs  = _load_msgs()
    users = _load(USERS_FILE)

    # Determine which users this person is allowed to message (enrollment-based)
    if user.get("role") == "doctor":
        # Doctor sees only their approved enrolled patients
        doctor_record  = users.get(uid, {})
        allowed_ids    = set(doctor_record.get("patient_list", []))
    else:
        # Patient sees only their assigned (approved) doctor
        patient_record = users.get(uid, {})
        assigned       = patient_record.get("assigned_doctor_id")
        allowed_ids    = {assigned} if assigned else set()

    partners = {}
    # First: seed all allowed contacts so they always appear (even with no messages)
    for other_id in allowed_ids:
        u = users.get(other_id, {})
        if u:
            partners[other_id] = {
                "user_id":   other_id,
                "full_name": u.get("full_name", "Unknown"),
                "role":      u.get("role", "patient"),
                "last_msg":  None,
                "last_ts":   "",
            }

    # Then: layer in actual message data for allowed partners
    for m in msgs:
        if uid not in m.get("deleted_by", []):
            if m["sender_id"] == uid:
                other = m["recipient_id"]
            elif m["recipient_id"] == uid:
                other = m["sender_id"]
            else:
                continue
            # Only show messages with allowed partners
            if other and other != uid and other in allowed_ids:
                if other not in partners or m["timestamp"] > partners[other].get("last_ts", ""):
                    u = users.get(other, {})
                    partners[other] = {
                        "user_id":   other,
                        "full_name": u.get("full_name", "Unknown"),
                        "role":      u.get("role", "patient"),
                        "last_msg":  m["text"],
                        "last_ts":   m["timestamp"],
                    }

    convs = sorted(partners.values(), key=lambda x: x["last_ts"], reverse=True)
    return {"conversations": convs}


# ── GET /api/messages/{other_user_id}  (DYNAMIC — after static routes) ───────
@router.get("/messages/{other_user_id}")
def get_messages(other_user_id: str, authorization: str = Header(...)):
    user = _auth(authorization)
    uid  = user["id"]
    msgs = _load_msgs()
    changed = False
    # Mark all incoming messages from this conversation partner as read.
    for m in msgs:
        if (m["sender_id"] == other_user_id and m["recipient_id"] == uid):
            rb = m.setdefault("read_by", [])
            if uid not in rb:
                rb.append(uid)
                changed = True
    if changed:
        _save_msgs(msgs)

    conv = [
        m for m in msgs
        if ((m["sender_id"] == uid    and m["recipient_id"] == other_user_id) or
            (m["sender_id"] == other_user_id and m["recipient_id"] == uid))
        and uid not in m.get("deleted_by", [])
    ]
    return {"messages": conv}


# ── DELETE /api/messages/{message_id} ────────────────────────────────────────
@router.delete("/messages/{message_id}")
def delete_message(message_id: str, authorization: str = Header(...)):
    user = _auth(authorization)
    msgs = _load_msgs()
    for m in msgs:
        if m["id"] == message_id:
            if user["id"] not in m.get("deleted_by", []):
                m.setdefault("deleted_by", []).append(user["id"])
            break
    _save_msgs(msgs)
    return {"ok": True}


