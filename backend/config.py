import os

# ── Scoring weights ───────────────────────────────────────────────────────────
# Must sum to 1.0
SPEECH_WEIGHT   = float(os.getenv("SPEECH_WEIGHT",   "0.4"))
MEMORY_WEIGHT   = float(os.getenv("MEMORY_WEIGHT",   "0.4"))
REACTION_WEIGHT = float(os.getenv("REACTION_WEIGHT", "0.2"))

assert abs(SPEECH_WEIGHT + MEMORY_WEIGHT + REACTION_WEIGHT - 1.0) < 1e-6, \
    "Scoring weights must sum to 1.0"

# ── Risk thresholds (applied to composite score 0–100) ────────────────────────
# score >= THRESHOLD_LOW  → "Low" risk
# score >= THRESHOLD_HIGH → "Moderate" risk
# score <  THRESHOLD_HIGH → "High" risk
THRESHOLD_LOW  = float(os.getenv("THRESHOLD_LOW",  "70"))
THRESHOLD_HIGH = float(os.getenv("THRESHOLD_HIGH", "40"))

# ── Model paths (placeholders for future ML integration) ──────────────────────
SPEECH_MODEL_PATH   = os.getenv("SPEECH_MODEL_PATH",   "models/weights/speech_model.onnx")
MEMORY_MODEL_PATH   = os.getenv("MEMORY_MODEL_PATH",   "models/weights/memory_model.onnx")
REACTION_MODEL_PATH = os.getenv("REACTION_MODEL_PATH", "models/weights/reaction_model.onnx")

# ── Server ────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
DEBUG    = os.getenv("DEBUG", "true").lower() == "true"

# ── CORS ──────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
