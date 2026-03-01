import statistics
from typing import Optional


def compute_trend(scores: list[float]) -> str:
    """
    Compute linear trend direction over a list of scores.
    Returns: 'improving' | 'declining' | 'stable' | 'insufficient_data'
    """
    if len(scores) < 2:
        return "insufficient_data"

    # Simple linear regression slope
    n = len(scores)
    x_mean = (n - 1) / 2
    y_mean = statistics.mean(scores)

    numerator   = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(scores))
    denominator = sum((i - x_mean) ** 2 for i in range(n))

    if denominator == 0:
        return "stable"

    slope = numerator / denominator

    if slope > 1.0:      return "improving"
    elif slope < -1.0:   return "declining"
    else:                return "stable"


def compute_change_rate(scores: list[float]) -> Optional[float]:
    """
    Compute percentage change from first recorded score to latest.
    Returns None if insufficient data.
    """
    if len(scores) < 2:
        return None
    first  = scores[0]
    latest = scores[-1]
    if first == 0:
        return None
    return round(((latest - first) / first) * 100, 2)


def build_progress_summary(historical_results: list[dict]) -> dict:
    """
    Build a full progress summary from historical results.

    historical_results: list of result dicts in chronological order (oldest first)
    Returns per-metric trends + overall health trajectory.
    """
    if not historical_results:
        return {
            "session_count": 0,
            "overall_trajectory": "no_data",
            "metrics": {},
        }

    METRICS = [
        ("memory_score",    "Memory"),
        ("speech_score",    "Speech"),
        ("reaction_score",  "Reaction Time"),
        ("executive_score", "Executive Function"),
        ("motor_score",     "Motor Control"),
    ]

    metric_summaries = {}
    trajectory_scores = []

    for field, label in METRICS:
        series = [r[field] for r in historical_results if field in r]
        if not series:
            continue

        metric_summaries[field] = {
            "label":        label,
            "latest":       round(series[-1], 2),
            "average":      round(statistics.mean(series), 2),
            "best":         round(max(series), 2),
            "worst":        round(min(series), 2),
            "trend":        compute_trend(series),
            "change_rate":  compute_change_rate(series),
            "history":      [round(s, 2) for s in series],
        }
        trajectory_scores.append(compute_trend(series))

    # Overall trajectory = most common trend
    if trajectory_scores:
        improving_count = trajectory_scores.count("improving")
        declining_count = trajectory_scores.count("declining")
        if declining_count > improving_count:
            overall = "declining"
        elif improving_count > declining_count:
            overall = "improving"
        else:
            overall = "stable"
    else:
        overall = "insufficient_data"

    # Risk trend from stored probabilities
    risk_fields = ["alzheimers_risk", "dementia_risk", "parkinsons_risk"]
    risk_trends = {}
    for rf in risk_fields:
        series = [r[rf] for r in historical_results if rf in r]
        if series:
            risk_trends[rf] = {
                "latest":  round(series[-1], 4),
                "average": round(statistics.mean(series), 4),
                "trend":   compute_trend([s * 100 for s in series]),
            }

    return {
        "session_count":      len(historical_results),
        "overall_trajectory": overall,
        "metrics":            metric_summaries,
        "risk_trends":        risk_trends,
    }
