from __future__ import annotations

from typing import Optional, Dict, Any, Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Leaflore Brain API", version="1.1.0")

# CORS for Lovable + web clients
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

    # Optional context for ANY subject/grade
    student_id: Optional[str] = None
    board: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
    concept: Optional[str] = None
    language: Optional[str] = "english"

    # optional “signals” from UI/analytics
    signals: Optional[Dict[str, Any]] = None


class RespondResponse(BaseModel):
    text: str
    mode: str = "teach"
    next_action: str = "clarify"
    ui_hint: str = "Ask a question or press Speak"
    memory_update: Dict[str, Any] = {}


# ----------------------------
# Helpers
# ----------------------------
def _clean(s: Optional[str]) -> str:
    return (s or "").strip()


def _topic(ctx: RespondRequest) -> str:
    # Prefer most specific
    return _clean(ctx.concept) or _clean(ctx.chapter) or _clean(ctx.subject) or "this topic"


def _detect_intent(text: str) -> str:
    t = text.lower().strip()

    if any(x in t for x in ["your name", "who are you", "teacher name", "what's your name", "whats your name"]):
        return "intro"
    if any(x in t for x in ["confuse", "confused", "don't understand", "cant understand", "can’t understand"]):
        return "confused"
    if any(x in t for x in ["scared", "worried", "anxious", "panic", "stress", "stressed", "nervous"]):
        return "anxious"

    if any(x in t for x in ["define", "meaning of", "what is", "what are", "explain"]):
        return "explain"
    if any(x in t for x in ["example", "real life", "analogy"]):
        return "example"
    if any(x in t for x in ["steps", "how to", "process", "method"]):
        return "steps"
    if any(x in t for x in ["difference", "compare", "vs", "versus"]):
        return "compare"
    if any(x in t for x in ["solve", "calculate", "find", "answer", "equation", "sum", "problem"]):
        return "solve"

    return "general"


def _neuro_style_opening(intent: str) -> str:
    if intent == "anxious":
        return "Hi 🙂 You’re safe here. Thanks for telling me."
    return "Hi 🙂 Thanks for sharing that."


def _grade_subject_line(ctx: RespondRequest) -> str:
    g = _clean(ctx.grade)
    s = _clean(ctx.subject)
    if g and s:
        return f"Grade {g} • {s}"
    if g:
        return f"Grade {g}"
    if s:
        return s
    return ""


def _reply(student_text: str, ctx: RespondRequest) -> str:
    intent = _detect_intent(student_text)
    opener = _neuro_style_opening(intent)

    topic = _topic(ctx)
    meta = _grade_subject_line(ctx)
    meta_line = f"{meta}\n" if meta else ""

    # Gentle neuro-support line (non-clinical, calm)
    support = ""
    if intent == "anxious":
        support = "Before we start: take one slow breath. We’ll do this step-by-step.\n\n"

    # Intro
    if intent == "intro":
        return (
            f"{opener}\n\n"
            "I’m Anaya — your friendly teacher. I’ll teach in a calm way and help you feel confident.\n\n"
            "What name should I call you?"
        )

    # Confused
    if intent == "confused":
        return (
            f"{opener}\n\n"
            f"{support}"
            f"{meta_line}"
            f"No problem — confusion is part of learning.\n"
            f"Tell me the *exact line/word* that feels confusing about **{topic}**, and I’ll explain it with a simple example."
        )

    # Explain / Define
    if intent == "explain":
        return (
            f"{opener}\n\n"
            f"{support}"
            f"{meta_line}"
            f"Let’s understand **{topic}** clearly.\n"
            "To explain it perfectly, choose one:\n"
            "1) Easy meaning\n"
            "2) Step-by-step\n"
            "3) Real-life example\n\n"
            "Reply with 1, 2, or 3."
        )

    # Example
    if intent == "example":
        return (
            f"{opener}\n\n"
            f"{support}"
            f"{meta_line}"
            f"Sure — I’ll give a simple real-life example for **{topic}**.\n\n"
            "Quick check: do you want a *school-level* example or a *daily-life* example?"
        )

    # Steps / Process
    if intent == "steps":
        return (
            f"{opener}\n\n"
            f"{support}"
            f"{meta_line}"
            f"Okay — we’ll do **{topic}** step-by-step.\n"
            "First, tell me what you have:\n"
            "• the question text (or)\n"
            "• the part you reached\n\n"
            "Paste it here."
        )

    # Compare
    if intent == "compare":
        return (
            f"{opener}\n\n"
            f"{support}"
            f"{meta_line}"
            "Great — comparisons make learning fast.\n"
            "Tell me the two things you want to compare (A vs B), and I’ll explain:\n"
            "• Meaning\n"
            "• Key differences\n"
            "• One simple example"
        )

    # Solve (universal: math/science/logic/grammar etc.)
    if intent == "solve":
        return (
            f"{opener}\n\n"
            f"{support}"
            f"{meta_line}"
            "I can help you solve it calmly.\n"
            "Please paste the full question.\n\n"
            "Also tell me: do you want the *final answer only* or *step-by-step with explanation*?"
        )

    # Default (universal)
    return (
        f"{opener}\n\n"
        f"{support}"
        f"{meta_line}"
        f"Let’s work on **{topic}**.\n"
        "Tell me what you want:\n"
        "• Meaning (easy)\n"
        "• Explanation\n"
        "• Example\n"
        "• Practice question\n\n"
        "Reply with one of these words."
    )


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"service": "Leaflore Brain API", "health": "/health", "respond": "/respond"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/respond", response_model=RespondResponse)
def respond(payload: RespondRequest) -> RespondResponse:
    if payload.action != "respond":
        return RespondResponse(
            text="Please send action='respond' with your question.",
            mode="teach",
            next_action="clarify",
            ui_hint="Try again",
            memory_update={},
        )

    text = _reply(payload.student_input, payload)

    # Minimal memory update example (optional)
    memory_update: Dict[str, Any] = {}
    if payload.subject:
        memory_update["last_subject"] = payload.subject
    if payload.grade:
        memory_update["last_grade"] = payload.grade

    return RespondResponse(
        text=text,
        mode="teach",
        next_action="clarify",
        ui_hint="Ask another question or press Speak",
        memory_update=memory_update,
    )
