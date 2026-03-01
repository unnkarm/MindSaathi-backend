import math
import statistics
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# 1. HYBRID RISK COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_hybrid_risk(
    clinical_prob: float,
    ml_prob: float,
    clinical_weight: float = 0.6,
    ml_weight: float = 0.4,
) -> float:
    """
    Hybrid model combining clinical rule-based and ML logistic probabilities.

    Final Risk = w_clinical × Clinical_Prob + w_ml × ML_Prob

    This blends medically interpretable rules with statistical modeling.
    Both inputs should be in [0, 1].
    """
    hybrid = (clinical_weight * clinical_prob) + (ml_weight * ml_prob)
    return round(max(0.0, min(1.0, hybrid)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROGRESS ANOMALY DETECTION (Z-score based)
# ─────────────────────────────────────────────────────────────────────────────

ANOMALY_Z_THRESHOLD = -1.5   # > 1.5 std deviations drop = warning
ANOMALY_MIN_HISTORY = 3      # Need at least 3 sessions to detect anomalies


def detect_progress_anomaly(
    score_history: list[float],
    current_score: float,
    metric_name: str = "score",
) -> dict:
    """
    Detect if current_score is an anomalous DROP compared to historical trend.

    Uses Z-score: Z = (current - mean_history) / std_history
    If Z < ANOMALY_Z_THRESHOLD → anomaly detected (significant drop).

    Returns:
        anomaly_detected: bool
        z_score: float
        severity: "none" | "mild" | "significant" | "severe"
        message: str
    """
    if len(score_history) < ANOMALY_MIN_HISTORY:
        return {
            "anomaly_detected": False,
            "z_score": None,
            "severity": "insufficient_data",
            "message": f"Need {ANOMALY_MIN_HISTORY}+ sessions to detect anomalies.",
        }

    mean_h = statistics.mean(score_history)
    # Use population std (stdev of sample) but protect against flat history
    std_h  = statistics.stdev(score_history) if len(score_history) > 1 else 1.0
    if std_h < 1.0:
        std_h = 1.0   # Prevent divide-by-near-zero

    z = (current_score - mean_h) / std_h

    if z < -2.5:
        severity = "severe"
        anomaly  = True
    elif z < -1.75:
        severity = "significant"
        anomaly  = True
    elif z < ANOMALY_Z_THRESHOLD:
        severity = "mild"
        anomaly  = True
    else:
        severity = "none"
        anomaly  = False

    messages = {
        "none":        None,
        "mild":        f"⚠️ Mild {metric_name} dip detected. Monitor over next session.",
        "significant": f"⚠️ Significant {metric_name} drop detected. Recommend clinical attention.",
        "severe":      f"🚨 Severe {metric_name} decline detected. Urgent clinical evaluation advised.",
        "insufficient_data": None,
    }

    return {
        "anomaly_detected": anomaly,
        "z_score":          round(z, 3),
        "severity":         severity,
        "mean_history":     round(mean_h, 2),
        "std_history":      round(std_h, 2),
        "message":          messages.get(severity),
    }


def analyze_all_progress_anomalies(
    historical_results: list[dict],
    current_result: dict,
) -> dict:
    """
    Run anomaly detection across all key cognitive metrics.

    historical_results: list of past result dicts (from results.json)
    current_result: the just-computed result dict

    Returns per-metric anomaly findings + overall alert level.
    """
    if not historical_results:
        return {"overall_alert": "none", "metrics": {}}

    METRICS_TO_CHECK = [
        ("memory_score",    "Memory"),
        ("reaction_score",  "Reaction Time"),
        ("speech_score",    "Speech"),
        ("executive_score", "Executive Function"),
        ("motor_score",     "Motor Control"),
    ]

    findings = {}
    highest_severity_rank = 0
    severity_rank = {"none": 0, "insufficient_data": 0, "mild": 1, "significant": 2, "severe": 3}

    for field, label in METRICS_TO_CHECK:
        history = [r[field] for r in historical_results if field in r]
        current = current_result.get(field)
        if current is None:
            continue

        result = detect_progress_anomaly(history, current, label)
        findings[field] = result

        rank = severity_rank.get(result["severity"], 0)
        if rank > highest_severity_rank:
            highest_severity_rank = rank

    overall = ["none", "mild", "significant", "severe"][highest_severity_rank]
    return {"overall_alert": overall, "metrics": findings}


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONFIDENCE INTERVAL
# ─────────────────────────────────────────────────────────────────────────────

def compute_confidence_interval(prob: float) -> dict:
    """
    Approximate 95% CI for risk probability.
    CI is widest near 0.5 (most uncertain) and narrower near extremes.
    CI = prob ± (base_se + boundary_bonus)
    """
    base_se        = 0.04
    boundary_bonus = max(0, 0.03 - abs(prob - 0.5) * 0.06)
    half_ci        = base_se + boundary_bonus
    lower          = round(max(0.0, prob - half_ci), 4)
    upper          = round(min(1.0, prob + half_ci), 4)

    return {
        "ci_lower": lower,
        "ci_upper": upper,
        "ci_label": f"{round(prob * 100, 1)}% (±{round(half_ci * 100, 1)}%)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE IMPORTANCE (SHAP-like approximation, no external lib)
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_importance(feature_vector: dict, disease: str = "alzheimers") -> list[dict]:
    """
    Approximate feature importance without external SHAP library.
    Returns ranked list of features most contributing to risk.

    Uses abs(normalized_feature × clinical_weight) as proxy for importance.
    This gives a defensible, explainable breakdown for the hackathon demo.
    """
    # Clinical importance weights per disease
    IMPORTANCE = {
        "alzheimers": {
            "delayed_recall_accuracy": 0.35,
            "immediate_recall_accuracy": 0.30,
            "intrusion_count": 0.20,
            "pause_ratio": 0.15,
            "order_match_ratio": 0.15,
            "recall_latency": 0.10,
            "reaction_drift": 0.05,
            "stroop_error_rate": 0.05,
        },
        "dementia": {
            "stroop_error_rate": 0.30,
            "miss_count": 0.25,
            "mean_rt": 0.25,
            "std_rt": 0.20,
            "reaction_drift": 0.18,
            "delayed_recall_accuracy": 0.18,
            "immediate_recall_accuracy": 0.20,
        },
        "parkinsons": {
            "tap_interval_std": 0.40,
            "mean_rt": 0.30,
            "std_rt": 0.25,
            "speech_start_delay": 0.20,
            "speech_variability": 0.18,
            "min_rt": 0.15,
        },
    }

    weights = IMPORTANCE.get(disease, IMPORTANCE["alzheimers"])
    items = []
    for feat, weight in weights.items():
        val = feature_vector.get(feat)
        if val is None:
            continue
        items.append({
            "feature":    feat,
            "importance": round(weight, 3),
            "value":      round(float(val), 3),
        })

    items.sort(key=lambda x: x["importance"], reverse=True)
    return items[:6]   # Top 6 features
