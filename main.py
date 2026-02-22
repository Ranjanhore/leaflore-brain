"""
main.py — Leaflore Brain (FastAPI)

What this file provides:
- ✅ FastAPI app at TOP LEVEL (Render-compatible)
- ✅ /health endpoint
- ✅ /respond endpoint that calls OpenAI with a SYSTEM PROMPT for "Leaflore Teacher"
- ✅ Returns STRICT JSON (teacher_text, phase, expects, ui)
- ✅ Keeps lightweight per-session memory using OpenAI Responses API `previous_response_id`
- ✅ Safe fallbacks if model returns non-JSON

Env vars you should set on Render:
- OPENAI_API_KEY (required)
- OPENAI_MODEL (optional, default: "gpt-5")
- ALLOWED_ORIGINS (optional, default: "*" meaning allow all)
"""

import json
import os
import re
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI


# ----------------------------
# Config
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = ["*"] if ALLOWED_ORIGINS.strip() == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]

if not OPENAI_API_KEY:
    # Fail fast at import time on Render so logs show the real cause.
    raise RuntimeError("Missing OPENAI_API_KEY environment variable")

client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------------------
# Leaflore Teacher System Prompt
# ----------------------------
SYSTEM_PROMPT = """
You are Leaflore Teacher (warm, patient, story-like). You are teaching a live class.

You must start every new class with:
1) Greeting + introduce yourself briefly.
2) Ask the student's name.
3) Ask one short warm-up question: grade/board or comfort level.
4) Explain class rules: when I'm explaining, stay quiet; if you want to interrupt use Raise Hand; I'll pause and let you speak.
5) Confirm readiness: “Ready? Say yes.”
Only after the student confirms, start the lesson.

You MUST respond ONLY as valid JSON with exactly this shape:

{
  "teacher_text": string,
  "phase": "INTRO" | "GET_NAME" | "WARMUP" | "RULES" | "READY_CHECK" | "TEACHING",
  "expects": "NONE" | "STUDENT_NAME" | "SHORT_ANSWER" | "YES_NO",
  "ui": { "mic": "OPEN" | "MUTED", "show_raise_hand": boolean }
}

Rules:
- teacher_text should be short, spoken-friendly, and friendly.
- During teacher explanation, set ui.mic="MUTED".
- When asking the student to respond, set ui.mic="OPEN" and show_raise_hand=true.
- Never include markdown. Never include extra keys. Never include trailing comments.
""".strip()


# ----------------------------
# Request/Response Models
# ----------------------------
EventType = Literal["class_start", "student_reply", "raise_hand", "ping"]

class RespondRequest(BaseModel):
    type: EventType = Field(default="student_reply")
    sessionId: str = Field(..., min_length=1)
    text: Optional[str] = None
    phase: Optional[str] = None
    student: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class TeacherUI(BaseModel):
    mic: Literal["OPEN", "MUTED"]
    show_raise_hand: bool


class TeacherResponse(BaseModel):
    teacher_text: str
    phase: Literal["INTRO", "GET_NAME", "WARMUP", "RULES", "READY_CHECK", "TEACHING"]
    expects: Literal["NONE", "STUDENT_NAME", "SHORT_ANSWER", "YES_NO"]
    ui: TeacherUI


# ----------------------------
# In-memory conversation state (per session)
# ----------------------------
# We keep only the last OpenAI response_id per session. This is enough to maintain conversation
# with Responses API using previous_response_id.
SESSION_STATE: Dict[str, Dict[str, Any]] = {}


# ----------------------------
# FastAPI app
# ----------------------------
app = FastAPI(title="Leaflore Brain", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


def _extract_first_json_object(s: str) -> Optional[str]:
    """
    If the model accidentally returns extra text, attempt to extract the first {...} block.
    """
    if not s:
        return None
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    # Greedy-ish extraction of the first JSON object
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    return m.group(0) if m else None


def _safe_parse_teacher_json(raw: str) -> Dict[str, Any]:
    """
    Parse model output into the expected TeacherResponse shape. If parsing fails,
    return a safe fallback JSON that won't break the client.
    """
    candidate = _extract_first_json_object(raw) or ""
    try:
        data = json.loads(candidate)
    except Exception:
        # Fallback response (client-safe)
        return {
            "teacher_text": "Hi! I’m your Leaflore teacher. What’s your name?",
            "phase": "GET_NAME",
            "expects": "STUDENT_NAME",
            "ui": {"mic": "OPEN", "show_raise_hand": True},
        }

    # Minimal validation + safe defaults
    if not isinstance(data, dict):
        return {
            "teacher_text": "Hi! I’m your Leaflore teacher. What’s your name?",
            "phase": "GET_NAME",
            "expects": "STUDENT_NAME",
            "ui": {"mic": "OPEN", "show_raise_hand": True},
        }

    teacher_text = str(data.get("teacher_text", "")).strip() or "Okay."
    phase = data.get("phase") or "INTRO"
    expects = data.get("expects") or "NONE"
    ui = data.get("ui") or {}
    mic = ui.get("mic") or "OPEN"
    show_raise_hand = bool(ui.get("show_raise_hand", True))

    # Clamp to allowed values
    phase_allowed = {"INTRO","GET_NAME","WARMUP","RULES","READY_CHECK","TEACHING"}
    expects_allowed = {"NONE","STUDENT_NAME","SHORT_ANSWER","YES_NO"}
    mic_allowed = {"OPEN","MUTED"}

    if phase not in phase_allowed:
        phase = "INTRO"
    if expects not in expects_allowed:
        expects = "NONE"
    if mic not in mic_allowed:
        mic = "OPEN"

    return {
        "teacher_text": teacher_text,
        "phase": phase,
        "expects": expects,
        "ui": {"mic": mic, "show_raise_hand": show_raise_hand},
    }


def _build_user_input(req: RespondRequest) -> str:
    """
    Convert your frontend event into a concise user input for the model.
    """
    student = req.student or {}
    ctx = req.context or {}

    # Keep it compact; model already has the full behavior in system prompt.
    payload = {
        "event": req.type,
        "sessionId": req.sessionId,
        "phase": req.phase,
        "text": req.text,
        "student": {
            "name": student.get("name"),
            "grade": student.get("grade"),
            "board": student.get("board"),
        },
        "context": ctx,
    }
    return json.dumps(payload, ensure_ascii=False)


@app.post("/respond")
def respond(req: RespondRequest) -> TeacherResponse:
    """
    Client calls this for:
    - class_start (Start button)
    - student_reply (speech-to-text transcript)
    - raise_hand (interrupt)
    """
    try:
        prev_id = SESSION_STATE.get(req.sessionId, {}).get("previous_response_id")

        # Responses API (recommended). Use text.format json_object to force valid JSON. :contentReference[oaicite:0]{index=0}
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=SYSTEM_PROMPT,
            input=_build_user_input(req),
            previous_response_id=prev_id,
            store=False,
            # Force JSON output
            text={"format": {"type": "json_object"}},
        )

        # Save conversation state
        SESSION_STATE[req.sessionId] = {"previous_response_id": response.id}

        raw = (response.output_text or "").strip()
        parsed = _safe_parse_teacher_json(raw)

        # Validate against Pydantic model (will raise if wrong shape)
        return TeacherResponse(**parsed)

    except Exception as e:
        # Return a client-safe error while keeping HTTP 200 optional.
        # Here we return 500 with safe message so you can debug quickly.
        raise HTTPException(status_code=500, detail=f"Brain error: {str(e)}")
