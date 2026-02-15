import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Leaflore Brain API", version="1.0.0")

# ✅ CORS: allow Lovable + local dev.
# For quick testing you can keep "*" but production is safer with explicit origins.
ALLOWED_ORIGINS = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "https://lovable.dev,https://*.lovable.app,http://localhost:5173,http://localhost:3000",
)

# FastAPI CORSMiddleware does not understand wildcard subdomains like https://*.lovable.app directly.
# So we allow any origin in middleware and enforce a light origin check below.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # we enforce in-code below
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _origin_allowed(origin: Optional[str]) -> bool:
    if not origin:
        return True  # non-browser clients
    allowed = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
    if origin in allowed:
        return True
    # allow lovable subdomains
    if origin.endswith(".lovable.app"):
        return True
    return False

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/respond")
async def respond(req: Request):
    origin = req.headers.get("origin")
    if not _origin_allowed(origin):
        return JSONResponse(
            status_code=403,
            content={"error": "Origin not allowed", "origin": origin},
        )

    payload: Dict[str, Any] = {}
    try:
        payload = await req.json()
    except Exception:
        payload = {}

    # Accept either student_input or text (be forgiving)
    student_input = (
        payload.get("student_input")
        or payload.get("text")
        or payload.get("message")
        or ""
    )

    action = payload.get("action", "respond")

    # Minimal "teacher" reply (replace this with your real model logic)
    # Keep response shape simple for the frontend: { "text": "..." }
    if not student_input or not isinstance(student_input, str):
        return JSONResponse(status_code=400, content={"error": "Missing student_input"})

    # Example teacher response (you can plug your pipeline here)
    chapter = payload.get("chapter") or "today's topic"
    grade = payload.get("grade") or "6"
    reply = (
        f"Hi! Grade {grade} Science time 😊 You said: “{student_input}”. "
        f"Let’s learn about {chapter}. What do you want to know first?"
    )

    return {
        "ok": True,
        "action": action,
        "text": reply,
        "meta": {
            "chapter": chapter,
            "grade": grade,
        },
    }

# Optional: root
@app.get("/")
def root():
    return {"service": "Leaflore Brain API", "health": "/health", "respond": "/respond"}
