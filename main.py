from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Leaflore Brain API", version="1.0.0")


# -----------------------------
# CORS (Lovable + local dev + general web)
# -----------------------------
# If you ever set allow_origins=["*"], then allow_credentials MUST be False.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Models
# -----------------------------
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
    language: Optional[str] = "english"
    signals: Optional[Dict[str, Any]] = None


class RespondResponse(BaseModel):
    text: str
    reply: str
    message: str
    meta: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------
# Helpers: "Human teacher" style
# -----------------------------
_SMALLTALK_PATTERNS = [
    r"\bhow are you\b",
    r"\bwhat'?s up\b",
    r"\bwhat is your name\b",
    r"\byour name\b",
    r"\bhave you eaten\b",
    r"\blunch\b",
    r"\bdinner\b",
    r"\bbreakfast\b",
    r"\bdo you like\b",
    r"\bwho are you\b",
]

_GREETINGS = re.compile(r"^\s*(hi|hello|hey|hii|hlo)\b", re.I)


def _is_smalltalk(text: str) -> bool:
    t = text.lower().strip()
    if _GREETINGS.match(t) and len(t.split()) <= 3:
        return True
    for p in _SMALLTALK_PATTERNS:
        if re.search(p, t, re.I):
            return True
    return False


def _looks_like_question(text: str) -> bool:
    t = text.strip()
    return ("?" in t) or any(t.lower().startswith(x) for x in ["what", "why", "how", "when", "where", "who", "can", "is", "are", "do", "does"])


def _neuro_warm_opening() -> str:
    # Keep it natural, not robotic. No repeated “I heard you say…”
    return "Got you 🙂"


def _reflect_emotion(signals: Optional[Dict[str, Any]]) -> str:
    if not signals:
        return ""
    emo = (signals.get("emotion") or "").lower().strip()
    if emo in ["sad", "upset", "anxious", "worried", "scared", "stressed"]:
        return "If you’re feeling a bit stressed, we’ll go slowly—no pressure. "
    if emo in ["angry", "frustrated"]:
        return "I can see this might feel frustrating—let’s break it into tiny steps. "
    if emo in ["happy", "excited"]:
        return "Love the energy—let’s use it well. "
    return ""


def _gentle_redirect_to_learning() -> str:
    return "Want to learn something today, or should we just chat for a minute?"


def _answer_smalltalk(student_text: str) -> str:
    t = student_text.lower()
    if "name" in t or "who are you" in t:
        return "I’m your Leaflore teacher—here with you like a friendly mentor. What would you like help with today?"
    if "lunch" in t or "eaten" in t or "dinner" in t or "breakfast" in t:
        return "Aww thanks for asking 😊 I had something light. How about you—did you eat? And what are we learning today?"
    if "how are you" in t or "what's up" in t:
        return "I’m doing good—happy to be here with you. What’s on your mind right now?"
    return "😊 I’m here with you. Tell me what you’d like to talk about or learn."


def _universal_teaching_reply(student_text: str) -> str:
    """
    A human-like teaching pattern:
    1) Warm acknowledge
    2) Answer simply
    3) Ask 1 follow-up question (like ChatGPT)
    """
    text = student_text.strip()

    # If it’s not even a question, invite clarity in a warm way
    if not _looks_like_question(text) and len(text.split()) < 4:
        return (
            "Okay 🙂 Tell me a little more—what exactly do you want to understand? "
            "You can say: meaning, steps, an example, or a quick quiz."
        )

    # If it is a question but we don’t have an LLM, we still respond naturally:
    # Keep it short + ask a follow up.
    return (
        "Let’s do this together. "
        "First, tell me one thing you already know about it (even a small guess). "
        "Then I’ll explain it in the easiest way and give you a quick example."
    )


def build_teacher_reply(req: RespondRequest) -> str:
    student_text = (req.student_input or "").strip()

    # Warm, emotionally aware teacher voice
    emo_line = _reflect_emotion(req.signals)

    if _is_smalltalk(student_text):
        # Pure small talk handling (human + friendly + gently redirects)
        return f"{emo_line}{_answer_smalltalk(student_text)}"

    # Universal teaching style (works for all subjects)
    opening = _neuro_warm_opening()
    core = _universal_teaching_reply(student_text)

    # Keep it natural (no “I heard you say…”)
    return f"{emo_line}{opening} {core}".strip()


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root() -> Dict[str, str]:
    return {"service": "Leaflore Brain API", "health": "/health", "respond": "/respond"}


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/respond")
def respond_help() -> Dict[str, Any]:
    return {
        "ok": True,
        "method": "POST",
        "example_body": {"action": "respond", "student_input": "What is photosynthesis?"},
        "note": "Use POST /respond with JSON body.",
    }


@app.options("/respond")
def respond_options() -> Dict[str, str]:
    # CORS preflight handled by middleware; this is just extra-safe.
    return {"status": "ok"}


@app.post("/respond", response_model=RespondResponse)
async def respond(req: RespondRequest, request: Request) -> RespondResponse:
    reply_text = build_teacher_reply(req)

    return RespondResponse(
        text=reply_text,
        reply=reply_text,
        message=reply_text,
        meta={
            "mode": "neuro-balanced-teacher",
            "universal": True,
            "subject_hint": req.subject,
            "grade_hint": req.grade,
        },
    )
