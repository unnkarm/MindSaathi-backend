# ── Scoring weights (must sum to 1.0) ─────────────────────────────────────────
DOMAIN_WEIGHTS = {
    "speech":    0.25,
    "memory":    0.30,
    "reaction":  0.20,
    "executive": 0.15,
    "motor":     0.10,
}

# ── Risk thresholds (probability scale 0–1) ────────────────────────────────────
THRESHOLDS = {
    "low_max":      0.35,
    "moderate_max": 0.65,
    # > 0.65 → High
}

# ── Age-bracket norms for z-score normalization ────────────────────────────────
AGE_NORMS = {
    "reaction_time": {   # ms — lower is better
        "20-39": {"mean": 280, "std": 45},
        "40-59": {"mean": 330, "std": 55},
        "60-75": {"mean": 400, "std": 70},
        "75+":   {"mean": 480, "std": 90},
    },
    "memory_accuracy": {  # % — higher is better
        "20-39": {"mean": 82, "std": 12},
        "40-59": {"mean": 75, "std": 13},
        "60-75": {"mean": 65, "std": 15},
        "75+":   {"mean": 55, "std": 18},
    },
    "wpm": {              # words per minute — higher is better
        "20-39": {"mean": 145, "std": 30},
        "40-59": {"mean": 135, "std": 30},
        "60-75": {"mean": 120, "std": 32},
        "75+":   {"mean": 105, "std": 35},
    },
}


def get_age_bracket(age: int) -> str:
    if age < 40:   return "20-39"
    elif age < 60: return "40-59"
    elif age < 75: return "60-75"
    else:          return "75+"


def age_z_score(value: float, metric: str, age: int) -> float:
    """Z = (X - μ_age) / σ_age. Positive = above peer mean (better)."""
    bracket = get_age_bracket(age)
    norms = AGE_NORMS.get(metric, {}).get(bracket)
    if not norms:
        return 0.0
    return round((value - norms["mean"]) / norms["std"], 3)


# ── Education correction (cognitive reserve) ───────────────────────────────────
# 1=No formal, 2=Primary, 3=Secondary, 4=Graduate, 5=Postgrad
EDUCATION_CORRECTION = {
    1: +0.05,
    2: +0.03,
    3:  0.00,
    4:  0.00,
    5: -0.02,
}


def get_education_correction(education_level: int) -> float:
    return EDUCATION_CORRECTION.get(education_level, 0.0)


# ── Medical condition risk multipliers (γ coefficients) ───────────────────────
CONDITION_MULTIPLIERS = {
    "diabetes":          0.04,
    "hypertension":      0.05,
    "stroke_history":    0.08,
    "family_alzheimers": 0.06,
    "parkinsons_dx":     0.10,
    "depression":        0.04,
    "thyroid_disorder":  0.03,
}

MAX_RISK_CAP = 0.95


def apply_condition_multipliers(base_risk: float, conditions: dict) -> float:
    """R_final = R × (1 + Σγ) capped at MAX_RISK_CAP."""
    gamma_sum = sum(
        CONDITION_MULTIPLIERS.get(k, 0.0)
        for k, v in conditions.items() if v
    )
    return min(base_risk * (1 + gamma_sum), MAX_RISK_CAP)


# ── Fatigue / temporary factor configuration ───────────────────────────────────
FATIGUE_FACTORS = {
    "tired":          0.10,
    "sleep_deprived": 0.12,
    "sick":           0.08,
    "anxious":        0.06,
}

FATIGUE_CONFIDENCE_THRESHOLD = 0.75


def compute_confidence_score(missing_data_ratio: float, fatigue_flags: dict) -> float:
    """Confidence = 1 - MissingDataRatio - FatiguePenalty. Range [0, 1]."""
    fatigue_penalty = sum(
        FATIGUE_FACTORS.get(k, 0.0)
        for k, v in fatigue_flags.items() if v
    )
    confidence = 1.0 - missing_data_ratio - fatigue_penalty
    return round(max(0.0, min(1.0, confidence)), 3)


# ── Safe output language ───────────────────────────────────────────────────────
SAFE_OUTPUT_LANGUAGE = {
    "disclaimer": (
        "⚠️ This is NOT a diagnosis. This tool identifies cognitive risk indicators only. "
        "Always consult a qualified neurologist or physician for clinical evaluation."
    ),
    "retest_recommendation": (
        "Results may be temporarily affected by fatigue or stress. "
        "Please retest after adequate rest for a more reliable reading."
    ),
}
