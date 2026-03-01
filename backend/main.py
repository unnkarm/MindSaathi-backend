from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routers import analyze, auth, messages, content, chat, games
from utils.logger import log_info

app = FastAPI(
    title="MindSaathi API",
    description="Backend for MindSaathi cognitive risk assessment",
    version="1.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(analyze.router,  prefix="/api")
app.include_router(auth.router,     prefix="/api")
app.include_router(messages.router, prefix="/api")
app.include_router(content.router,  prefix="/api")
app.include_router(chat.router,     prefix="/api")
app.include_router(games.router,    prefix="/api")

# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log_info(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "MindSaathi Backend"}
