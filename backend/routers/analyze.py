"""
analyze.py — MindSaathi V4
Full pipeline: 18-feature extraction → 3-disease logistic models →
V2 clinical layers → hybrid ML scoring → composite risk → risk_drivers →
anomaly detection → JSON persistence.
"""

import json, os, math
from datetime import datetime
from fastapi import APIRouter, HTTPException, Header
from typing import Optional

from models.schemas import AnalyzeRequest, AnalyzeResponse, DiseaseRiskLevels
from services.ai_service import (
    extract_speech_features, extract_memory_features,
    extract_reaction_features, extract_executive_features,
    extract_motor_features, compute_disease_risks,
    build_feature_vector, _prob_to_level,
)
from core.clinical_config import (
    apply_condition_multipliers, compute_confidence_score,
    get_education_correction, FATIGUE_CONFIDENCE_THRESHOLD,
    SAFE_OUTPUT_LANGUAGE, DOMAIN_WEIGHTS,
)
from core.ml_engine import (
    compute_hybrid_risk, compute_confidence_interval,
    analyze_all_progress_anomalies, compute_feature_importance,
)
from core.progress_tracker import build_progress_summary
from utils.logger import log_info

router = APIRouter()
DISCLAIMER = SAFE_OUTPUT_LANGUAGE["disclaimer"]

DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_FILE  = os.path.join(DATA_DIR, "results.json")
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")
USERS_FILE    = os.path.join(DATA_DIR, "users.json")
os.makedirs(DATA_DIR, exist_ok=True)


def _load(path):
    if not os.path.exists(path): return {}
    with open(path) as f:
        try: return json.load(f)
        except: return {}

def _save(path, data):
    with open(path, "w") as f: json.dump(data, f, indent=2)

def _user_from_token(token: str) -> Optional[dict]:
    sessions = _load(SESSIONS_FILE)
    session  = sessions.get(token)
    if not session: return None
    users = _load(USERS_FILE)
    return users.get(session["user_id"])


def _compute_composite_risk(
    speech: float, memory: float, reaction: float,
    executive: float, motor: float,
) -> float:
    """
    Composite risk score (0–100, higher = more risk).
    Inverts domain scores (higher domain score = healthier = lower risk).
    Uses DOMAIN_WEIGHTS for weighted average.
    """
    w = DOMAIN_WEIGHTS
    risk = (
        w["speech"]    * (100 - speech)
        + w["memory"]    * (100 - memory)
        + w["reaction"]  * (100 - reaction)
        + w["executive"] * (100 - executive)
        + w["motor"]     * (100 - motor)
    )
    return round(max(0.0, min(100.0, risk)), 2)


def _compute_risk_drivers(
    speech: float, memory: float, reaction: float,
    executive: float, motor: float,
) -> dict:
    """
    Risk contribution of each domain as percentage of total composite risk.
    Used by RiskDriversPanel on the frontend.
    """
    w = DOMAIN_WEIGHTS
    contributions = {
        "speech":    w["speech"]    * (100 - speech),
        "memory":    w["memory"]    * (100 - memory),
        "reaction":  w["reaction"]  * (100 - reaction),
        "executive": w["executive"] * (100 - executive),
        "motor":     w["motor"]     * (100 - motor),
    }
    total = sum(contributions.values()) or 1.0
    pcts = {k: round((v / total) * 100) for k, v in contributions.items()}

    # Map to frontend field names expected by RiskDriversPanel
    return {
        "memory_recall_contribution_pct":      pcts["memory"],
        "executive_function_contribution_pct": pcts["executive"],
        "speech_delay_contribution_pct":       pcts["speech"],
        "reaction_time_contribution_pct":      pcts["reaction"],
        "motor_consistency_contribution_pct":  pcts["motor"],
    }


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    payload: AnalyzeRequest,
    authorization: Optional[str] = Header(default=None),
):
    log_info(f"[/api/analyze] submitting full pipeline")

    try:
        speech_score,  sf  = extract_speech_features(payload.speech_audio or None, payload.speech)
        memory_score,  mf  = extract_memory_features(payload.memory_results, payload.memory)
        reaction_score, rf = extract_reaction_features(payload.reaction_times, payload.reaction)
        exec_score,    ef  = extract_executive_features(payload.stroop)
        motor_score,   mof = extract_motor_features(payload.tap)

        fv    = build_feature_vector(sf, mf, rf, ef, mof)
        risks = compute_disease_risks(fv, payload.profile)

        alz_risk  = risks["alzheimers_risk"]
        dem_risk  = risks["dementia_risk"]
        park_risk = risks["parkinsons_risk"]

        mean_rt = rf.get("mean_rt", 1)
        std_rt  = rf.get("std_rt", 0)
        avi     = round(std_rt / mean_rt, 4) if mean_rt > 0 else 0.0

    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Processing error: {exc}")

    # ── Clinical layers ────────────────────────────────────────────────────────
    conditions_dict = payload.conditions.model_dump() if payload.conditions else {}
    fatigue_dict    = payload.fatigue.model_dump()    if payload.fatigue    else {}

    alz_risk_adj  = apply_condition_multipliers(alz_risk,  conditions_dict)
    dem_risk_adj  = apply_condition_multipliers(dem_risk,  conditions_dict)
    park_risk_adj = apply_condition_multipliers(park_risk, conditions_dict)

    if payload.profile and payload.profile.education_level:
        edu_corr     = get_education_correction(payload.profile.education_level)
        memory_score = max(0.0, min(100.0, memory_score + edu_corr * 100))

    confidence       = compute_confidence_score(0.0, fatigue_dict)
    recommend_retest = confidence < FATIGUE_CONFIDENCE_THRESHOLD

    # ── Hybrid risk + CI ───────────────────────────────────────────────────────
    hybrid_risk = compute_hybrid_risk(alz_risk_adj, alz_risk)
    ci          = compute_confidence_interval(hybrid_risk)

    # ── Composite risk score (for ProgressPage wellness display) ──────────────
    composite_risk = _compute_composite_risk(
        speech_score, memory_score, reaction_score, exec_score, motor_score
    )

    # ── Risk drivers (for RiskDriversPanel explainability) ────────────────────
    risk_drivers = _compute_risk_drivers(
        speech_score, memory_score, reaction_score, exec_score, motor_score
    )

    # ── Feature importance ─────────────────────────────────────────────────────
    fv_dict            = fv.model_dump()
    feature_importance = compute_feature_importance(fv_dict, disease="alzheimers")

    # ── Model validation (simulated — for ValidationPanel) ────────────────────
    model_validation = {
        "sensitivity": 0.82,
        "specificity": 0.78,
        "auc":         0.85,
        "note":        "Simulated validation due to absence of clinical dataset.",
    }

    # ── Build result record ────────────────────────────────────────────────────
    result_data = {
        "timestamp":            datetime.utcnow().isoformat(),
        "createdAt":            datetime.utcnow().isoformat(),  # for ProgressPage
        "speech_score":         speech_score,
        "memory_score":         memory_score,
        "reaction_score":       reaction_score,
        "executive_score":      exec_score,
        "motor_score":          motor_score,
        "alzheimers_risk":      alz_risk_adj,
        "dementia_risk":        dem_risk_adj,
        "parkinsons_risk":      park_risk_adj,
        "composite_risk_score": composite_risk,
        "hybrid_risk":          hybrid_risk,
        "confidence":           confidence,
        "risk_levels": {
            "alzheimers": _prob_to_level(alz_risk_adj),
            "dementia":   _prob_to_level(dem_risk_adj),
            "parkinsons": _prob_to_level(park_risk_adj),
        },
        "attention_variability_index": avi,
        "disclaimer": DISCLAIMER,
    }

    # ── Anomaly detection + save ───────────────────────────────────────────────
    anomaly_result = {"overall_alert": "none", "metrics": {}}

    if authorization:
        token = authorization.replace("Bearer ", "").strip()
        user  = _user_from_token(token)
        if user:
            results = _load(RESULTS_FILE)
            uid     = user["id"]
            history = results.get(uid, [])
            anomaly_result = analyze_all_progress_anomalies(history, result_data)
            history.append(result_data)
            results[uid] = history[-20:]
            _save(RESULTS_FILE, results)

    return AnalyzeResponse(
        speech_score=speech_score,
        memory_score=memory_score,
        reaction_score=reaction_score,
        executive_score=exec_score,
        motor_score=motor_score,
        alzheimers_risk=alz_risk_adj,
        dementia_risk=dem_risk_adj,
        parkinsons_risk=park_risk_adj,
        risk_levels=DiseaseRiskLevels(
            alzheimers=_prob_to_level(alz_risk_adj),
            dementia=_prob_to_level(dem_risk_adj),
            parkinsons=_prob_to_level(park_risk_adj),
        ),
        composite_risk_score=composite_risk,
        hybrid_risk=hybrid_risk,
        confidence=confidence,
        recommend_retest=recommend_retest,
        ci_lower=ci["ci_lower"],
        ci_upper=ci["ci_upper"],
        ci_label=ci["ci_label"],
        logistic_risk_probability=alz_risk_adj,    # primary risk signal
        confidence_interval_label=ci["ci_label"],
        anomaly_alert=anomaly_result["overall_alert"],
        anomaly_details=anomaly_result["metrics"] if anomaly_result["overall_alert"] != "none" else None,
        risk_drivers=risk_drivers,
        feature_importance=feature_importance,
        model_validation=model_validation,
        feature_vector=fv,
        attention_variability_index=avi,
        disclaimer=DISCLAIMER,
    )


@router.get("/results/my")
def get_my_results(authorization: str = Header(...)):
    token = authorization.replace("Bearer ", "").strip()
    user  = _user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    results      = _load(RESULTS_FILE)
    user_results = results.get(user["id"], [])
    progress     = build_progress_summary(user_results)
    return {"results": user_results, "progress": progress}


@router.get("/results/patient/{patient_id}")
def get_patient_results(patient_id: str, authorization: str = Header(...)):
    token = authorization.replace("Bearer ", "").strip()
    user  = _user_from_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    if user.get("role", "patient") != "doctor":
        raise HTTPException(status_code=403, detail="Doctors only.")
    results         = _load(RESULTS_FILE)
    patient_results = results.get(patient_id, [])
    progress        = build_progress_summary(patient_results)
    return {"results": patient_results, "progress": progress}


