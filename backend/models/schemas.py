"""
schemas.py — MindSaathi V4
Extended with all fields required by ResultsPage, ProgressPage, and ProfileSetup.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ── Sub-payloads from frontend ─────────────────────────────────────────────────

class SpeechData(BaseModel):
    audio_b64: Optional[str] = None
    wpm: Optional[float] = None
    speed_deviation: Optional[float] = None
    speech_speed_variability: Optional[float] = None
    pause_ratio: Optional[float] = None
    completion_ratio: Optional[float] = None
    restart_count: Optional[int] = 0
    speech_start_delay: Optional[float] = None

class MemoryData(BaseModel):
    word_recall_accuracy: float = Field(default=50.0, ge=0, le=100)
    pattern_accuracy: float = Field(default=50.0, ge=0, le=100)
    delayed_recall_accuracy: Optional[float] = Field(default=None, ge=0, le=100)
    recall_latency_seconds: Optional[float] = None
    order_match_ratio: Optional[float] = None
    intrusion_count: Optional[int] = 0

class ReactionData(BaseModel):
    times: List[float] = []
    miss_count: Optional[int] = 0
    initiation_delay: Optional[float] = None

class StroopData(BaseModel):
    total_trials: int = 0
    error_count: int = 0
    mean_rt: Optional[float] = None
    incongruent_rt: Optional[float] = None

class TapData(BaseModel):
    intervals: List[float] = []
    tap_count: int = 0

class UserProfile(BaseModel):
    age: Optional[int] = None
    education_level: Optional[int] = None   # 1–5
    sleep_hours: Optional[float] = None

class MedicalConditions(BaseModel):
    diabetes: bool = False
    hypertension: bool = False
    stroke_history: bool = False
    family_alzheimers: bool = False
    parkinsons_dx: bool = False
    depression: bool = False
    thyroid_disorder: bool = False

class FatigueFlags(BaseModel):
    tired: bool = False
    sleep_deprived: bool = False
    sick: bool = False
    anxious: bool = False

# ── Main request ───────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    speech_audio: Optional[str] = None
    memory_results: Dict[str, float] = {"word_recall_accuracy": 50.0, "pattern_accuracy": 50.0}
    reaction_times: List[float] = []
    speech: Optional[SpeechData] = None
    memory: Optional[MemoryData] = None
    reaction: Optional[ReactionData] = None
    stroop: Optional[StroopData] = None
    tap: Optional[TapData] = None
    profile: Optional[UserProfile] = None
    conditions: Optional[MedicalConditions] = None
    fatigue: Optional[FatigueFlags] = None

# ── Feature vector (18 features) ──────────────────────────────────────────────

class FeatureVector(BaseModel):
    wpm: float
    speed_deviation: float
    speech_variability: float
    pause_ratio: float
    speech_start_delay: float
    immediate_recall_accuracy: float
    delayed_recall_accuracy: float
    intrusion_count: float
    recall_latency: float
    order_match_ratio: float
    mean_rt: float
    std_rt: float
    min_rt: float
    reaction_drift: float
    miss_count: float
    stroop_error_rate: float
    stroop_rt: float
    tap_interval_std: float

class DiseaseRiskLevels(BaseModel):
    alzheimers: str
    dementia: str
    parkinsons: str

# ── Response (V4) ──────────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    # Domain scores (0–100, higher = healthier)
    speech_score: float
    memory_score: float
    reaction_score: float
    executive_score: float
    motor_score: float

    # Disease-specific probabilities (0–1)
    alzheimers_risk: float
    dementia_risk: float
    parkinsons_risk: float
    risk_levels: DiseaseRiskLevels

    # V4 composite + wellness
    composite_risk_score: Optional[float] = None   # 0–100, higher = more risk

    # V4 hybrid + CI
    hybrid_risk: Optional[float] = None
    confidence: Optional[float] = None
    recommend_retest: Optional[bool] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    ci_label: Optional[str] = None
    logistic_risk_probability: Optional[float] = None
    confidence_interval_label: Optional[str] = None

    # V4 anomaly detection
    anomaly_alert: Optional[str] = None
    anomaly_details: Optional[Dict[str, Any]] = None

    # V4 explainability — risk_drivers (for RiskDriversPanel)
    risk_drivers: Optional[Dict[str, float]] = None

    # V4 feature importance
    feature_importance: Optional[List[Dict]] = None

    # V4 model validation
    model_validation: Optional[Dict[str, Any]] = None

    # Feature transparency
    feature_vector: Optional[FeatureVector] = None
    attention_variability_index: Optional[float] = None

    disclaimer: str = (
        "⚠️ This is a behavioral screening tool only. "
        "It is NOT a medical diagnosis. Always consult a qualified "
        "neurologist or physician for clinical evaluation."
    )


