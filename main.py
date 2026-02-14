# main.py
import os
import base64
import logging
from typing import Any, Dict, Optional, List

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("leaflore-brain")


# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Leaflore Brain API", version="1.0.0")


def _parse_origins(val: str) -> List[str]:
    items = [x.strip() for x in (val or "").split(",")]
    return [x for x in items if x]


DEFAULT_ORIGINS = [
    "https://lovable.dev",
    "https://*.lovable.app",
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]
ALLOWED_ORIGINS = _parse_origins(os.getenv("ALLOWED_ORIGINS", "")) or DEFAULT_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Models
# ----------------------------
class RespondRequest(BaseModel):
    action: str = Field(default="respond")
    student_input: str = Field(..., min_length=1)

    # Session + meta (optional but recommended)
    session_id: Optional[str] = None
    board: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
    concept: Optional[str] = None
    language: Optional[str] = "english"

    # Optional signals the frontend might send
    signals: Optional[Dict[str, Any]] = None


class RespondResponse(BaseModel):
    session_id: Optional[str] = None
    text: str
    audio_base64: Optional[str] = None
    audio_mime: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


# ----------------------------
# Health
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------
# Teacher reply (LLM placeholder)
# ----------------------------
def generate_teacher_reply(req: RespondRequest) -> str:
    """
    Production-safe baseline reply generator.
    Replace this with your own LLM logic later if needed.
    """
    student = req.student_input.strip()
    chapter = (req.chapter or "today's topic").strip()
    grade = (req.grade or "").strip()
    subject = (req.subject or "Science").strip()

    # Simple onboarding / demo style
    if "introduce" in student.lower() or "start demo" in student.lower() or "start class" in student.lower():
        return (
            f"Hello! I’m Anaya, your {subject} teacher.\n\n"
            f"Welcome to our demo class. Before we begin, what’s your name?\n\n"
            "Quick class rule: when I’m explaining, please don’t speak. "
            "If you have a question, use the Raise Hand button and I’ll pause for you."
        )

    # Friendly, grade-aware response
    grade_line = f"Grade {grade} " if grade else ""
    return (
        f"Hi! {grade_line}{subject} time 😊\n\n"
        f"You said: “{student}”.\n\n"
        f"Today we’re learning about **{chapter}**.\n"
        "Tell me one thing you already know about it, or ask me a question (for example: "
        "“What is chlorophyll?”)."
    )


# ----------------------------
# ElevenLabs TTS (optional)
# ----------------------------
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")  # set this to your preferred voice id
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")

# If you want to disable TTS even when keys exist, set ENABLE_TTS=false
ENABLE_TTS = os.getenv("ENABLE_TTS", "true").lower() in ("1", "true", "yes", "y")


async def elevenlabs_tts_base64(text: str) -> Optional[Dict[str, str]]:
    """
    Returns {"audio_base64": "...", "audio_mime": "audio/mpeg"} or None if not configured.
    """
    if not (ENABLE_TTS and ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID):
        return None

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.4,
            "similarity_boost": 0.85,
        },
    }

    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            logger.warning("ElevenLabs TTS error %s: %s", r.status_code, r.text[:300])
            return None

        audio_bytes = r.content
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {"audio_base64": audio_b64, "audio_mime": "audio/mpeg"}


# ----------------------------
# Main endpoint
# ----------------------------
@app.post("/respond", response_model=RespondResponse)
async def respond(req: RespondRequest):
    if (req.action or "").strip().lower() != "respond":
        raise HTTPException(status_code=400, detail="Invalid action. Expected 'respond'.")

    if not req.student_input or not req.student_input.strip():
        raise HTTPException(status_code=400, detail="student_input is required.")

    teacher_text = generate_teacher_reply(req)

    # Optional audio
    tts = await elevenlabs_tts_base64(teacher_text)

    return RespondResponse(
        session_id=req.session_id,
        text=teacher_text,
        audio_base64=(tts or {}).get("audio_base64"),
        audio_mime=(tts or {}).get("audio_mime"),
        debug={
            "tts_enabled": bool(ENABLE_TTS),
            "tts_configured": bool(ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID),
            "origins": ALLOWED_ORIGINS,
        },
    )
