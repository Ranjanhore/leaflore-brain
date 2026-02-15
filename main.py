from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Leaflore Brain API", version="1.0.0")


# ----------------------------
# CORS (Lovable + local dev + safe defaults)
# Notes:
# - If you set allow_origins=["*"], you MUST keep allow_credentials=False.
# - This setup fixes the "Origin not allowed" / 403 / preflight issues.
# ----------------------------
ALLOW_ORIGIN_REGEX = os.getenv(
    "ALLOW_ORIGIN_REGEX",
    r"^https://.*\.lovable(app|project)\.com$|^https://.*\.lovable\.app$|^http://localhost(:\d+)?$",
)

# Easiest option (works for most apps):
# allow_origins=["*"] + allow_credentials=False
# If you need cookies/auth later, switch to explicit origins + allow_credentials=True.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Models
# ----------------------------
class RespondRequest(BaseModel):
    action: str = Field(default="respond")
    student_input: str

    # Optional context (safe defaults)
    student_id: Optional[str] = None
    board: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
    concept: Optional[str] = None
    language: Optional[str] = None
    signals: Optional[Dict[str, Any]] = None


# ----------------------------
# Helpers
# ----------------------------
def _pick_topic(req: RespondRequest) -> str:
    return req.subject or req.chapter or req.concept or "general"


def _teacher_reply(req: RespondRequest) -> Dict[str, Any]:
    """
    Replace this with your real teacher brain (LLM / RAG / rules).
    Keeping it deterministic + safe for now.
    """
    text = req.student_input.strip()
    topic = _pick_topic(req)

    # Example: simple friendly teacher response
    reply_text = (
        f"Hi! I heard you say: '{text}'.\n\n"
        f"Let’s learn about **{topic}** together. "
        f"Tell me: what do you already know about it, or ask one specific question?"
    )

    return {
        "text": reply_text,
        "emotion": "encouraging",
        "topic": topic,
        "next_prompt": "Ask me one specific doubt (for example: 'What is chlorophyll?').",
        "meta": {
            "student_id": req.student_id,
            "grade": req.grade,
            "board": req.board,
            "language": req.language,
        },
    }


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"service": "Leaflore Brain API", "health": "/health", "respond": "/respond"}


@app.get("/health")
def health():
    return {"status": "ok"}


# Optional convenience endpoint (some people test /Render)
@app.get("/Render")
def render_health():
    return {"status": "ok"}


@app.post("/respond")
async def respond(payload: RespondRequest, request: Request):
    # Basic validation
    if payload.action != "respond":
        raise HTTPException(status_code=400, detail="Invalid action. Use action='respond'.")

    if not payload.student_input or not payload.student_input.strip():
        raise HTTPException(status_code=400, detail="student_input cannot be empty")

    try:
        return _teacher_reply(payload)
    except Exception as e:
        # Never crash the server with unhandled exceptions
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ----------------------------
# Local run (Render uses uvicorn main:app ...)
# ----------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
