"""
auth.py — MindSaathi Authentication Router
Handles register, login, logout with Firebase Firestore storage.
Role-separated: patients cannot login to doctor panel and vice versa.
"""

import re
import hashlib
import secrets
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, Field
from typing import Optional
from firebase_config import db

router = APIRouter(prefix="/auth", tags=["auth"])

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")

# ── Firestore helpers ─────────────────────────────────────────────────────────

def _get_users() -> dict:
    docs = db.collection("users").stream()
    return {doc.id: doc.to_dict() for doc in docs}

def _save_users(users: dict):
    for uid, user in users.items():
        db.collection("users").document(uid).set(user)

def _get_user_by_id(user_id: str) -> Optional[dict]:
    doc = db.collection("users").document(user_id).get()
    return doc.to_dict() if doc.exists else None

def _save_user(user_id: str, user: dict):
    db.collection("users").document(user_id).set(user)

def _get_sessions() -> dict:
    docs = db.collection("sessions").stream()
    return {doc.id: doc.to_dict() for doc in docs}

def _save_sessions(sessions: dict):
    for token, session in sessions.items():
        db.collection("sessions").document(token).set(session)

def _create_session(user_id: str) -> str:
    token = secrets.token_hex(32)
    db.collection("sessions").document(token).set({
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat(),
    })
    return token

def _delete_session(token: str):
    db.collection("sessions").document(token).delete()

def _get_user_from_token(token: str) -> Optional[dict]:
    session_doc = db.collection("sessions").document(token).get()
    if not session_doc.exists:
        return None
    session = session_doc.to_dict()
    user_id = session.get("user_id")
    if not user_id:
        return None
    user_doc = db.collection("users").document(user_id).get()
    return user_doc.to_dict() if user_doc.exists else None

def _hash_password(password: str) -> str:
    salted = f"neuroaid_salt_{password}"
    return hashlib.sha256(salted.encode()).hexdigest()

def _safe_user(user: dict) -> dict:
    return {k: v for k, v in user.items() if k != "password_hash"}


# ── Schemas ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., min_length=5)
    password: str = Field(..., min_length=6, max_length=128)
    role: str = Field(default="patient")
    age: Optional[int] = Field(default=None, ge=1, le=120)
    gender: Optional[str] = None
    phone: Optional[str] = None
    license_number: Optional[str] = None
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
    role: str = Field(default="patient")


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
    if not _EMAIL_RE.match(body.email.strip()):
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")

    if body.role not in ("patient", "doctor"):
        raise HTTPException(status_code=400, detail="Role must be 'patient' or 'doctor'.")

    # Check duplicate email within same role
    users = _get_users()
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

    if body.role == "doctor":
        new_user.update({
            "specialization": body.specialization,
            "hospital": body.hospital,
            "location": body.location,
            "years_experience": body.years_experience,
            "consultation_mode": body.consultation_mode or "Both",
            "bio": body.bio,
            "max_patients": body.max_patients or 10,
            "current_patients": 0,
            "patient_list": [],
            "pending_requests": [],
        })

    _save_user(user_id, new_user)
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

    if matched_user.get("role", "patient") != body.role:
        if body.role == "doctor":
            raise HTTPException(status_code=403, detail="This account is registered as a Patient. Please use the Patient panel.")
        else:
            raise HTTPException(status_code=403, detail="This account is registered as a Doctor. Please use the Doctor panel.")

    # Update last login
    matched_user["last_login"] = datetime.utcnow().isoformat()
    _save_user(matched_id, matched_user)

    token = _create_session(matched_id)
    return AuthResponse(message="Login successful!", token=token, user=_safe_user(matched_user))


@router.post("/logout")
def logout(authorization: str = Header(...)):
    """Logout — invalidates the session token."""
    token = authorization.replace("Bearer ", "").strip()
    session_doc = db.collection("sessions").document(token).get()
    if not session_doc.exists:
        raise HTTPException(status_code=401, detail="Invalid or expired session.")
    _delete_session(token)
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
    """Doctors only — get all registered patients."""
    token = authorization.replace("Bearer ", "").strip()
    user = _get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized. Please log in.")
    if user.get("role", "patient") != "doctor":
        raise HTTPException(status_code=403, detail="Access denied. Doctors only.")

    users = _get_users()
    doctor_data = users.get(user["id"], {})
    enrolled_ids = set(doctor_data.get("patient_list", []))

    # Load results
    results_map = {}
    results_docs = db.collection("results").stream()
    for doc in results_docs:
        results_map[doc.id] = doc.to_dict().get("results", [])

    patients = []
    for u in users.values():
        if u.get("role", "patient") != "patient":
            continue
        if u["id"] not in enrolled_ids:
            continue
        p = _safe_user(u)
        uid = u["id"]
        user_results = results_map.get(uid, [])
        p["sessionCount"] = len(user_results)
        p["lastResult"] = user_results[-1] if user_results else None
        patients.append(p)

    patients.sort(key=lambda p: p.get("last_login", ""), reverse=True)
    return {"patients": patients}


@router.put("/me")
def update_profile(body: UserProfileUpdate, authorization: str = Header(...)):
    """Update the logged-in user's profile."""
    token = authorization.replace("Bearer ", "").strip()
    user = _get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized. Please log in.")

    user_id = user["id"]
    if body.full_name is not None:
        user["full_name"] = body.full_name
    if body.age is not None:
        user["age"] = body.age
    if body.gender is not None:
        user["gender"] = body.gender
    if body.phone is not None:
        user["phone"] = body.phone

    _save_user(user_id, user)
    return {"message": "Profile updated.", "user": _safe_user(user)}


@router.put("/profile-extended")
def update_profile_extended(body: dict, authorization: str = Header(...)):
    """Save all extended patient clinical profile fields."""
    token = authorization.replace("Bearer ", "").strip()
    user = _get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized.")

    CLINICAL_FIELDS = [
        "age", "phone", "gender", "handedness", "education", "occupation",
        "medicalHistory", "currentMeds", "priorHeadInjury", "exerciseFreq",
        "smokingStatus", "alcoholUse", "sleepHours", "sleepQuality",
        "depressionHistory", "anxietyHistory", "familyHistory", "familyHistoryDetails",
        "existingDiagnosis", "cognitiveComplaints", "baselineTestDate",
    ]
    for field in CLINICAL_FIELDS:
        if field in body:
            user[field] = body[field]

    _save_user(user["id"], user)
    return {"message": "Extended profile saved.", "user": _safe_user(user)}


@router.get("/doctors")
def get_doctors(authorization: str = Header(...)):
    """Patients — get all registered doctors."""
    token = authorization.replace("Bearer ", "").strip()
    user = _get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized.")

    users = _get_users()
    doctors = []
    for u in users.values():
        if u.get("role") == "doctor":
            d = _safe_user(u)
            d["current_patients"] = len(u.get("patient_list", []))
            d["max_patients"] = u.get("max_patients", 10)
            doctors.append(d)
    return {"doctors": doctors}


@router.post("/doctors/enroll")
def enroll_with_doctor(body: dict, authorization: str = Header(...)):
    """Patient requests enrollment with a doctor."""
    token = authorization.replace("Bearer ", "").strip()
    user = _get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    if user.get("role") == "doctor":
        raise HTTPException(status_code=400, detail="Doctors cannot enroll with doctors.")

    doctor_id = body.get("doctor_id")
    if not doctor_id:
        raise HTTPException(status_code=400, detail="doctor_id required.")

    doctor_doc = db.collection("users").document(doctor_id).get()
    if not doctor_doc.exists:
        raise HTTPException(status_code=404, detail="Doctor not found.")
    doctor = doctor_doc.to_dict()
    if doctor.get("role") != "doctor":
        raise HTTPException(status_code=404, detail="Doctor not found.")

    approved = len(doctor.get("patient_list", []))
    max_p = doctor.get("max_patients", 10)
    if approved >= max_p:
        raise HTTPException(status_code=400, detail="This doctor has reached maximum patient capacity.")

    if user["id"] in doctor.get("patient_list", []):
        raise HTTPException(status_code=400, detail="You are already enrolled with this doctor.")

    pending = doctor.get("pending_requests", [])
    if any(r["patient_id"] == user["id"] for r in pending):
        raise HTTPException(status_code=400, detail="Your enrollment request is already pending.")

    pending.append({
        "patient_id": user["id"],
        "patient_name": user["full_name"],
        "patient_email": user["email"],
        "requested_at": datetime.utcnow().isoformat(),
    })
    doctor["pending_requests"] = pending
    _save_user(doctor_id, doctor)

    # Mark on patient side
    user["pending_doctor_id"] = doctor_id
    _save_user(user["id"], user)

    return {"message": "Enrollment request sent. Waiting for doctor approval.", "doctor": _safe_user(doctor)}


@router.post("/doctors/approve")
def approve_patient(body: dict, authorization: str = Header(...)):
    """Doctor approves or rejects a patient enrollment request."""
    token = authorization.replace("Bearer ", "").strip()
    doctor = _get_user_from_token(token)
    if not doctor or doctor.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Doctors only.")

    patient_id = body.get("patient_id")
    action = body.get("action")
    if not patient_id or action not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="patient_id and action ('approve'/'reject') required.")

    doc_id = doctor["id"]
    doctor_doc = db.collection("users").document(doc_id).get().to_dict()

    # Remove from pending
    pending = doctor_doc.get("pending_requests", [])
    doctor_doc["pending_requests"] = [r for r in pending if r["patient_id"] != patient_id]

    if action == "approve":
        if "patient_list" not in doctor_doc:
            doctor_doc["patient_list"] = []
        if patient_id not in doctor_doc["patient_list"]:
            doctor_doc["patient_list"].append(patient_id)
        doctor_doc["current_patients"] = len(doctor_doc["patient_list"])
        _save_user(doc_id, doctor_doc)

        patient_doc = db.collection("users").document(patient_id).get()
        if patient_doc.exists:
            patient = patient_doc.to_dict()
            patient["assigned_doctor_id"] = doc_id
            patient.pop("pending_doctor_id", None)
            _save_user(patient_id, patient)

    elif action == "reject":
        _save_user(doc_id, doctor_doc)
        patient_doc = db.collection("users").document(patient_id).get()
        if patient_doc.exists:
            patient = patient_doc.to_dict()
            patient.pop("pending_doctor_id", None)
            _save_user(patient_id, patient)

    verb = "approved" if action == "approve" else "rejected"
    return {"message": f"Patient {verb} successfully."}


@router.get("/doctors/my-doctor")
def get_my_doctor(authorization: str = Header(...)):
    """Patient — get their assigned doctor and pending status."""
    token = authorization.replace("Bearer ", "").strip()
    user = _get_user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized.")

    assigned_id = user.get("assigned_doctor_id")
    pending_id = user.get("pending_doctor_id")
    result = {"doctor": None, "pending_doctor": None}

    if assigned_id:
        doc = db.collection("users").document(assigned_id).get()
        if doc.exists:
            d = _safe_user(doc.to_dict())
            d["current_patients"] = len(doc.to_dict().get("patient_list", []))
            result["doctor"] = d

    if pending_id:
        doc = db.collection("users").document(pending_id).get()
        if doc.exists:
            result["pending_doctor"] = _safe_user(doc.to_dict())

    return result


@router.get("/doctors/pending-requests")
def get_pending_requests(authorization: str = Header(...)):
    """Doctor — get list of pending enrollment requests."""
    token = authorization.replace("Bearer ", "").strip()
    user = _get_user_from_token(token)
    if not user or user.get("role") != "doctor":
        raise HTTPException(status_code=403, detail="Doctors only.")

    doctor_doc = db.collection("users").document(user["id"]).get()
    if not doctor_doc.exists:
        return {"pending_requests": []}
    pending = doctor_doc.to_dict().get("pending_requests", [])
    return {"pending_requests": pending}