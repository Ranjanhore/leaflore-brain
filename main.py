from __future__ import annotations

from typing import Optional, Dict, Any, Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Leaflore Brain API", version="1.0.0")

# CORS: allow Lovable preview/publish + local dev
# (Wildcard is simplest; keep allow_credentials False with "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Models
# ----------------------------
class RespondRequest(BaseModel):
    action: Literal["respond"] = "respond"
    student_input: str = Field(..., min_length=1)

    # optional context fields (safe defaults)
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
    next_action: str = "clarify"
    ui_hint: str = "Ask a question or press Speak"
    memory_update: Dict[str, Any] = {}


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"service": "Leaflore Brain API", "health": "/health", "respond": "/respond"}


@app.get("/health")
def health():
    return {"status": "ok"}


def _neuro_teacher_reply(student_text: str, ctx: RespondRequest) -> str:
    """
    Neuro-therapist + balanced teacher style:
    - humble + reassuring
    - specific + short
    - asks ONE gentle follow-up question
    - validates emotions without being clinical
    """
    s = student_text.strip()

    # Light intent detection (simple, reliable, no external AI)
    lower = s.lower()
    is_confused = any(w in lower for w in ["confuse", "confused", "don't understand", "cant understand", "can’t understand"])
    is_anxious = any(w in lower for w in ["scared", "worried", "anxious", "panic", "stress", "stressed", "nervous"])
    asks_class = any(w in lower for w in ["which class", "what class", "today", "learn today", "we learn"])
    asks_name = any(w in lower for w in ["your name", "who are you", "teacher name", "what's your name", "whats your name"])

    # Pull context (keep it subtle)
    grade = (ctx.grade or "").strip()
    subject = (ctx.subject or "").strip()
    chapter = (ctx.chapter or "").strip()
    concept = (ctx.concept or "").strip()

    # Build a calm, warm opener
    opener = "Hi 🙂 Thanks for sharing that."
    if is_anxious:
        opener = "Hi 🙂 You’re safe here. Thanks for telling me."

    # Specific replies
    if asks_name:
        return (
            f"{opener}\n\n"
            "I’m Anaya — your friendly science teacher, and I’ll also help you learn in a calm, confident way.\n\n"
            "Before we start, what name should I call you?"
        )

    if asks_class:
        topic = chapter or concept or "today’s science topic"
        grade_part = f"Grade {grade} " if grade else ""
        subject_part = f"{subject} " if subject else ""
        return (
            f"{opener}\n\n"
            f"Today we’ll do {grade_part}{subject_part}and focus on **{topic}**.\n"
            "I’ll explain step-by-step, and you can ask anytime.\n\n"
            "Quick check: what do you already know about it (even 1 small point)?"
        )

    if is_confused:
        topic = chapter or concept or "this topic"
        return (
            f"{opener}\n\n"
            f"No problem — getting confused is part of learning. Let’s make **{topic}** easy.\n"
            "Tell me the *exact line/word* that feels confusing, and I’ll explain it with a simple example."
        )

    # Default: balanced teaching + gentle neuro-support
    topic = chapter or concept or "this"
    extra = ""
    if is_anxious:
        extra = "If you feel stuck, take one slow breath — we’ll do it together.\n"

    return (
        f"{opener}\n\n"
        f"{extra}"
        f"Let’s work on **{topic}**.\n"
        "Answer in one sentence:\n"
        "What exactly do you want to understand — meaning, steps, or an example?"
    )


@app.post("/respond", response_model=RespondResponse)
def respond(payload: RespondRequest) -> RespondResponse:
    # Only supported action for now
    if payload.action != "respond":
        return RespondResponse(
            text="I can help! Please send action='respond' with your question.",
            mode="teach",
            next_action="clarify",
            ui_hint="Try again",
            memory_update={},
        )

    reply_text = _neuro_teacher_reply(payload.student_input, payload)

    return RespondResponse(
        text=reply_text,
        mode="teach",
        next_action="clarify",
        ui_hint="Ask another question or press Speak",
        memory_update={},
    )
