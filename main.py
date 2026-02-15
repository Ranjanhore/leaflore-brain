from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Leaflore Brain API", version="1.0.0")


# -----------------------------
# CORS (IMPORTANT for Lovable)
# -----------------------------
# Lovable preview/publish often runs on domains like:
# - https://<something>.lovableproject.com
# Also allow localhost for dev.
ALLOW_ORIGIN_REGEX = os.getenv(
    "ALLOW_ORIGIN_REGEX",
    r"^https://.*\.lovableproject\.com$|^https://.*\.lovable\.dev$|^http://localhost(:\d+)?$",
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Models
# -----------------------------
class RespondRequest(BaseModel):
    action: str = Field(default="respond")
    student_input: str

    # Optional context fields (safe defaults)
    student_id: Optional[str] = None
    board: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
    concept: Optional[str] = None
    language: Optional[str] = "english"
    signals: Optional[Dict[str, Any]] = None


class RespondResponse(BaseModel):
    text: str
    mode: str = "teach"
    next_action: str = "clarify_actor"
    micro_q: str = "What do you want to learn next?"
    ui_hint: str = "Ask a short question like “What is chlorophyll?”"
    memory_update: Dict[str, Any] = {}


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    # This is what you saw working in your screenshot (root shows available routes)
    return {"service": "Leaflore Brain API", "health": "/health", "respond": "/respond"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/respond", response_model=RespondResponse)
async def respond(payload: RespondRequest, request: Request):
    # Basic validation (keep strict so frontend knows what's wrong)
    if payload.action != "respond":
        return RespondResponse(
            text=f"I only support action='respond'. You sent: {payload.action}",
            next_action="fix_request",
            micro_q="Please resend with action='respond'.",
        )

    student_text = (payload.student_input or "").strip()
    if not student_text:
        return RespondResponse(
            text="Please type something so I can help you 🙂",
            next_action="await_input",
            micro_q="Try: “What is photosynthesis?”",
        )

    # --- Teacher reply logic (replace with your real AI later) ---
    # For now, it gives a friendly teacher-style response + a follow-up question.
    topic = payload.chapter or payload.concept or payload.subject or "today’s topic"

    teacher_text = (
        f"Hi! I’m your teacher. You said: “{student_text}”.\n\n"
        f"Let’s learn about **{topic}**.\n"
        f"Tell me: what exactly is confusing you—meaning, steps, or an example?"
    )

    return RespondResponse(
        text=teacher_text,
        mode="teach",
        next_action="clarify_actor",
        micro_q="Do you want a simple definition or a real-life example?",
        ui_hint="Ask one short question at a time.",
        memory_update={
            "last_student_input": student_text,
            "topic": topic,
            "origin": request.headers.get("origin"),
        },
    )
