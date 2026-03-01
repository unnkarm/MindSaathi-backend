"""
auth.py — MindSaathi Authentication Router
Handles register, login, logout with local JSON file storage.
Role-separated: patients cannot login to doctor panel and vice versa.
"""

import json
import os
import re
import hashlib
import secrets
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, Field
from typing import Optional, List

router = APIRouter(prefix="/auth", tags=["auth"])

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")

# ── Local storage path ────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")

os.makedirs(DATA_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save_json(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _hash_password(password: str) -> str:
    salted = f"neuroaid_salt_{password}"
    return hashlib.sha256(salted.encode()).hexdigest()


def _get_users() -> dict:
    return _load_json(USERS_FILE)


def _save_users(users: dict):
    _save_json(USERS_FILE, users)


def _get_sessions() -> dict:
    return _load_json(SESSIONS_FILE)


def _save_sessions(sessions: dict):
    _save_json(SESSIONS_FILE, sessions)


def _create_session(user_id: str) -> str:
    token = secrets.token_hex(32)
    sessions = _get_sessions()
    sessions[token] = {
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat(),
    }
    _save_sessions(sessions)
    return token


def _get_user_from_token(token: str) -> Optional[dict]:
    sessions = _get_sessions()
    session = sessions.get(token)
    if not session:
        return None
    users = _get_users()
    return users.get(session["user_id"])


def _safe_user(user: dict) -> dict:
    return {k: v for k, v in user.items() if k != "password_hash"}


# ── Schemas ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., min_length=5)
    password: str = Field(..., min_length=6, max_length=128)
    role: str = Field(default="patient")          # "patient" or "doctor"
    age: Optional[int] = Field(default=None, ge=1, le=120)
    gender: Optional[str] = None
    phone: Optional[str] = None
    license_number: Optional[str] = None          # for doctors
    # Doctor-specific profile fields
    specialization: Optional[str] = None
    hospital: Optional[str] = None
    location: Optional[str] = None
    years_experience: Optional[int] = None
    consultation_mode: Optional[str] = None
    bio: Optional[str] = None
    max_patients: Optional[int] = 10


class LoginRequest(BaseModel):
    email: str
    password: str
    role: str = Field(default="patient")          # "patient" or "doctor"


class AuthResponse(BaseModel):
    message: str
    token: str
    user: dict


class UserProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse)
def register(body: RegisterRequest):
    """Register a new user (patient or doctor)."""
    # Validate email format
    if not _EMAIL_RE.match(body.email.strip()):
        raise HTTPException(status_code=400, detail="Please enter a valid email address (e.g. name@example.com).")

    # Validate role
    if body.role not in ("patient", "doctor"):
        raise HTTPException(status_code=400, detail="Role must be 'patient' or 'doctor'.")

    users = _get_users()

    # Check duplicate email within the same role
    for uid, user in users.items():
        if user["email"].lower() == body.email.lower() and user.get("role", "patient") == body.role:
            raise HTTPException(status_code=400, detail="Email already registered for this role.")

    user_id = str(uuid.uuid4())
    new_user = {
        "id": user_id,
        "full_name": body.full_name,
        "email": body.email.lower(),
        "password_hash": _hash_password(body.password),
        "role": body.role,
        "age": body.age,
        "gender": body.gender,
        "phone": body.phone,
        "license_number": body.license_number if body.role == "doctor" else None,
        "created_at": datetime.utcnow().isoformat(),
        "last_login": datetime.utcnow().isoformat(),
    }
    # Doctor-specific profile
    if body.role == "doctor":
        new_user.update({
            "specialization":   body.specialization,
            "hospital":         body.hospital,
            "location":         body.location,
            "years_experience": body.years_experience,
            "consultation_mode": body.consultation_mode or "Both",
            "bio":              body.bio,
            "max_patients":     body.max_patients or 10,
            "current_patients": 0,
            "patient_list":     [],
            "pending_requests": [],
        })

    users[user_id] = new_user
    _save_users(users)

    token = _create_session(user_id)
    return AuthResponse(message="Registration successful!", token=token, user=_safe_user(new_user))


@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest):
    """Login — role must match the panel the user is trying to access."""
    if body.role not in ("patient", "doctor"):
        raise HTTPException(status_code=400, detail="Role must be 'patient' or 'doctor'.")

    users = _get_users()

    matched_user = None
    matched_id = None
    for uid, user in users.items():
        if user["email"].lower() == body.email.lower():
            matched_user = user
            matched_id = uid
            break

    if not matched_user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    if matched_user["password_hash"] != _hash_password(body.password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    # ── Role guard: block cross-panel login ───────────────────────────────────
    if matched_user.get("role", "patient") != body.role:
        if body.role == "doctor":
            raise HTTPException(
                status_code=403,
                detail="This account is registered as a Patient. Please use the Patient panel."
            )
        else:
            raise HTTPException(
                status_code=403,
                detail="This account is registered as a Doctor. Please use the Doctor panel."
            )

    # Update last login
    users[matched_id]["last_login"] = datetime.utcnow().isoformat()
    _save_users(users)

    token = _create_session(matched_id)
    return AuthResponse(message="Login successful!", token=token, user=_safe_user(matched_user))


@router.post("/logout")
def logout(authorization: str = Header(...)):
    """Logout — invalidates the session token."""
    token = authorization.replace("Bearer ", "").strip()
    sessions = _get_sessions()

    if token not in sessions:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")

    del sessions[token]
    _save_sessions(sessions)
    return {"message": "Logged out successfully."}


@router.get("/me")
def get_current_user(authorization: str = Header(...)):
    """Get the currently logged-in user's profile."""
    token = authorization.replace("Bearer ", "").strip()
    user = _get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized. Please log in.")
    return {"user": _safe_user(user)}


@router.get("/patients")
def get_patients(authorization: str = Header(...)):
    """Doctors only — get all registered patients with their latest assessment result."""
    token = authorization.replace("Bearer ", "").strip()
    user = _get_user_from_token(token)

    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized. Please log in.")
    if user.get("role", "patient") != "doctor":
        raise HTTPException(status_code=403, detail="Access denied. Doctors only.")

    # Load results file
    results_path = os.path.join(DATA_DIR, "results.json")
    all_results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            try: all_results = json.load(f)
            except: all_results = {}

    users = _get_users()
    doctor_data    = users.get(user["id"], {})
    enrolled_ids   = set(doctor_data.get("patient_list", []))   # only approved patients

    patients = []
    for u in users.values():
        if u.get("role", "patient") != "patient":
            continue
        if u["id"] not in enrolled_ids:          # ← FILTER: only this doctor's patients
            continue
        p = _safe_user(u)
        uid = u["id"]
        user_results = all_results.get(uid, [])
        p["sessionCount"] = len(user_results)
        p["lastResult"]   = user_results[-1] if user_results else None
        patients.append(p)

    # Sort by last_login descending
    patients.sort(key=lambda p: p.get("last_login", ""), reverse=True)
    return {"patients": patients}


@router.put("/me")
def update_profile(body: UserProfileUpdate, authorization: str = Header(...)):
    """Update the logged-in user's profile."""
    token = authorization.replace("Bearer ", "").strip()
    sessions = _get_sessions()
    session = sessions.get(token)

    if not session:
        raise HTTPException(status_code=401, detail="Unauthorized. Please log in.")

    users = _get_users()
    user_id = session["user_id"]

    if body.full_name is not None:
        users[user_id]["full_name"] = body.full_name
    if body.age is not None:
        users[user_id]["age"] = body.age
    if body.gender is not None:
        users[user_id]["gender"] = body.gender
    if body.phone is not None:
        users[user_id]["phone"] = body.phone

    _save_users(users)
    return {"message": "Profile updated.", "user": _safe_user(users[user_id])}


@router.put("/profile-extended")
def update_profile_extended(body: dict, authorization: str = Header(...)):
    """Save all extended patient clinical profile fields."""
    token = authorization.replace("Bearer ", "").strip()
    user  = _get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    users = _get_users()
    uid   = user["id"]
    # Merge all clinical fields
    CLINICAL_FIELDS = [
        "age","phone","gender","handedness","education","occupation",
        "medicalHistory","currentMeds","priorHeadInjury","exerciseFreq",
        "smokingStatus","alcoholUse","sleepHours","sleepQuality",
        "depressionHistory","anxietyHistory","familyHistory","familyHistoryDetails",
        "existingDiagnosis","cognitiveComplaints","baselineTestDate",
    ]
    for field in CLINICAL_FIELDS:
        if field in body:
            users[uid][field] = body[field]
    _save_users(users)
    return {"message": "Extended profile saved.", "user": _safe_user(users[uid])}
    return {"message": "Profile updated.", "user": _safe_user(users[user_id])}


@router.get("/doctors")
def get_doctors(authorization: str = Header(...)):
    """Patients — get all registered doctors to start a conversation."""
    token = authorization.replace("Bearer ", "").strip()
    user  = _get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    users = _get_users()
    doctors = []
    for u in users.values():
        if u.get("role") == "doctor":
            d = _safe_user(u)
            d["current_patients"] = len(u.get("patient_list", []))
            d["max_patients"]     = u.get("max_patients", 10)
            doctors.append(d)
    return {"doctors": doctors}

@router.post("/doctors/enroll")
def enroll_with_doctor(body: dict, authorization: str = Header(...)):
    """Patient requests enrollment with a doctor (goes to pending, not auto-approved)."""
    token = authorization.replace("Bearer ", "").strip()
    user  = _get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    if user.get("role") == "doctor":
        raise HTTPException(status_code=400, detail="Doctors cannot enroll with doctors.")

    doctor_id = body.get("doctor_id")
    if not doctor_id:
        raise HTTPException(status_code=400, detail="doctor_id required.")

    users  = _get_users()
    doctor = users.get(doctor_id)
    if not doctor or doctor.get("role") != "doctor":
        raise HTTPException(status_code=404, detail="Doctor not found.")

    # Check capacity against approved patients
    approved = len(doctor.get("patient_list", []))
    max_p    = doctor.get("max_patients", 10)
    if approved >= max_p:
        raise HTTPException(status_code=400, detail="This doctor has reached maximum patient capacity.")

    # Check if already enrolled or pending
    if user["id"] in doctor.get("patient_list", []):
        raise HTTPException(status_code=400, detail="You are already enrolled with this doctor.")

    pending = doctor.get("pending_requests", [])
    if any(r["patient_id"] == user["id"] for r in pending):
        raise HTTPException(status_code=400, detail="Your enrollment request is already pending.")

    # Add to pending requests
    if "pending_requests" not in doctor:
        doctor["pending_requests"] = []
    doctor["pending_requests"].append({
        "patient_id": user["id"],
        "patient_name": user["full_name"],
        "patient_email": user["email"],
        "requested_at": datetime.utcnow().isoformat(),
    })
    users[doctor_id] = doctor

    # Mark on patient side
    users[user["id"]]["pending_doctor_id"] = doctor_id
    _save_users(users)

    return {"message": "Enrollment request sent. Waiting for doctor approval.", "doctor": _safe_user(doctor)}


@router.post("/doctors/approve")
def approve_patient(body: dict, authorization: str = Header(...)):
    """Doctor approves or rejects a patient enrollment request."""
    token = authorization.replace("Bearer ", "").strip()
    doctor = _get_user_from_token(token)
    if not doctor or doctor.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Doctors only.")

    patient_id = body.get("patient_id")
    action     = body.get("action")  # "approve" or "reject"
    if not patient_id or action not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="patient_id and action ('approve'/'reject') required.")

    users  = _get_users()
    doc_id = doctor["id"]

    # Remove from pending
    pending = users[doc_id].get("pending_requests", [])
    users[doc_id]["pending_requests"] = [r for r in pending if r["patient_id"] != patient_id]

    if action == "approve":
        if "patient_list" not in users[doc_id]:
            users[doc_id]["patient_list"] = []
        if patient_id not in users[doc_id]["patient_list"]:
            users[doc_id]["patient_list"].append(patient_id)
        users[doc_id]["current_patients"] = len(users[doc_id]["patient_list"])

        # Update patient
        if patient_id in users:
            users[patient_id]["assigned_doctor_id"] = doc_id
            users[patient_id].pop("pending_doctor_id", None)

    elif action == "reject":
        if patient_id in users:
            users[patient_id].pop("pending_doctor_id", None)

    _save_users(users)
    verb = "approved" if action == "approve" else "rejected"
    return {"message": f"Patient {verb} successfully."}


@router.get("/doctors/my-doctor")
def get_my_doctor(authorization: str = Header(...)):
    """Patient — get their assigned doctor and pending status."""
    token = authorization.replace("Bearer ", "").strip()
    user  = _get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized.")

    assigned_id = user.get("assigned_doctor_id")
    pending_id  = user.get("pending_doctor_id")
    users = _get_users()

    result = {"doctor": None, "pending_doctor": None}

    if assigned_id and assigned_id in users:
        d = _safe_user(users[assigned_id])
        d["current_patients"] = len(users[assigned_id].get("patient_list", []))
        result["doctor"] = d

    if pending_id and pending_id in users:
        d = _safe_user(users[pending_id])
        result["pending_doctor"] = d

    return result


@router.get("/doctors/pending-requests")
def get_pending_requests(authorization: str = Header(...)):
    """Doctor — get list of pending enrollment requests."""
    token = authorization.replace("Bearer ", "").strip()
    user  = _get_user_from_token(token)
    if not user or user.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Doctors only.")
    users    = _get_users()
    pending  = users[user["id"]].get("pending_requests", [])
    return {"pending_requests": pending}


