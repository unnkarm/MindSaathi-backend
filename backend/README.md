# MindSaathi — Backend

FastAPI backend for the **MindSaathi** cognitive risk assessment platform.  
It exposes a single `/api/analyze` endpoint that accepts multimodal data
(speech audio, memory game results, reaction times) and returns a structured
risk score. All scoring logic is currently **dummy/mock** — designed for
rapid frontend integration before the ML models are wired in.

---

## Folder Structure

```
backend/
├── main.py                  # FastAPI app entry point
├── config.py                # Weights, thresholds, model paths
├── requirements.txt
├── README.md
├── routers/
│   └── analyze.py           # POST /api/analyze
├── services/
│   └── ai_service.py        # Feature extraction + risk scoring
├── utils/
│   └── logger.py            # Logging helpers
└── models/
    └── schemas.py           # Pydantic request / response models
```

---

## Quickstart

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## Endpoints

### `GET /health`
Returns `{"status": "ok"}` — useful for Docker health checks.

---

### `POST /api/analyze`

**Request body** (`application/json`):

```json
{
  "speech_audio": "<base64-encoded-audio-string>",
  "memory_results": {
    "word_recall_accuracy": 80,
    "pattern_accuracy": 70
  },
  "reaction_times": [300, 280, 350, 310]
}
```

All fields are optional — defaults are applied when omitted so you can
test the endpoint with a minimal payload.

**Response** (`200 OK`):

```json
{
  "speech_score": 78.5,
  "memory_score": 75.0,
  "reaction_score": 82.3,
  "risk_score": 77.66,
  "risk_level": "Low"
}
```

| Field            | Type   | Description                                      |
|------------------|--------|--------------------------------------------------|
| `speech_score`   | float  | Speech health score (0–100, higher = healthier)  |
| `memory_score`   | float  | Memory performance score (0–100)                 |
| `reaction_score` | float  | Reaction speed score (0–100)                     |
| `risk_score`     | float  | Weighted composite: 0.4·speech + 0.4·memory + 0.2·reaction |
| `risk_level`     | string | `Low` (≥70) · `Moderate` (40–69) · `High` (<40) |

---

## Example cURL Request

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "memory_results": {"word_recall_accuracy": 80, "pattern_accuracy": 70},
    "reaction_times": [300, 280, 350, 310]
  }'
```

---

## Example JavaScript (React) Fetch

```js
const response = await fetch("http://localhost:8000/api/analyze", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    speech_audio: audioBase64,          // optional
    memory_results: {
      word_recall_accuracy: 80,
      pattern_accuracy: 70,
    },
    reaction_times: [300, 280, 350, 310],
  }),
});

const data = await response.json();
console.log(data.risk_level);           // "Low" | "Moderate" | "High"
```

---

## Configuration

All settings live in `config.py` and can be overridden with environment variables:

| Variable           | Default | Description                        |
|--------------------|---------|------------------------------------|
| `SPEECH_WEIGHT`    | `0.4`   | Weight for speech in risk formula  |
| `MEMORY_WEIGHT`    | `0.4`   | Weight for memory in risk formula  |
| `REACTION_WEIGHT`  | `0.2`   | Weight for reaction in risk formula|
| `THRESHOLD_LOW`    | `70`    | Score ≥ this → "Low" risk          |
| `THRESHOLD_HIGH`   | `40`    | Score ≥ this → "Moderate" risk     |
| `API_PORT`         | `8000`  | Server port                        |
| `DEBUG`            | `true`  | Enable debug mode                  |

---

## Replacing Dummy Logic

Each function in `services/ai_service.py` has a clear docstring explaining
what it should do. To plug in real ML models:

1. Load your model in `config.py` (path already configured).
2. Replace the function body in `ai_service.py` with inference code.
3. No changes needed in `routers/` or `main.py`.


