import os
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your lovable URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app = FastAPI()

# 🔥 IMPORTANT: allow your frontend origin
origins = [
    "https://lovable.dev",
    "http://localhost:5173",  # for local dev if needed
    "*",  # temporary for testing (can remove later)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "service": "Leaflore Brain API",
        "health": "/health",
        "respond": "/respond",
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/respond")
async def respond(data: dict):
    student_input = data.get("student_input", "")

    return {
        "text": f"Teacher says: I received your message: {student_input}"
    }

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
