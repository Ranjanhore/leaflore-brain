# main.py
import os
import re
import time
from typing import Optional, Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="Leaflore Brain", version="3.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://leaf-lore-chapters-story.lovable.app",
        "https://*.lovable.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────────────────────
# Supabase
# ─────────────────────────────────────────────────────────────

SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or os.getenv("VITE_SUPABASE_ANON_KEY")
)

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL / SUPABASE_KEY env vars")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ─────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────

ActionType = Literal["start_class", "respond", "next", "answer_quiz"]

class RespondPayload(BaseModel):
    action: ActionType
    student_id: str
    session_id: str
    chapter_id: str
    student_input: Optional[str] = None
    quiz_answer: Optional[str] = None


# ─────────────────────────────────────────────────────────────
# Teacher Profile
# ─────────────────────────────────────────────────────────────

DEFAULT_TEACHER = {
    "teacher_name": "Anaya Ma'am",
    "max_paragraph_words": 65,
}

def _get_teacher_profile(chapter_id: str) -> Dict[str, Any]:
    try:
        res = (
            supabase.table("chapters")
            .select("teacher_profile,teacher_name,voice_id")
            .eq("id", chapter_id)
            .maybe_single()
            .execute()
        )
        meta = res.data or {}
    except Exception:
        meta = {}

    profile = dict(DEFAULT_TEACHER)
    if isinstance(meta.get("teacher_profile"), dict):
        profile.update(meta["teacher_profile"])

    if meta.get("teacher_name"):
        profile["teacher_name"] = meta["teacher_name"]

    if meta.get("voice_id"):
        profile["voice_id"] = meta["voice_id"]

    return profile


# ─────────────────────────────────────────────────────────────
# Chunk Helpers
# ─────────────────────────────────────────────────────────────

def _load_chunks(chapter_id: str) -> List[Dict[str, Any]]:
    res = (
        supabase.table("chapter_chunks")
        .select("id,chapter_id,seq,type,title,chunk_text,media_url,duration_sec,quiz,is_active")
        .eq("chapter_id", chapter_id)
        .eq("is_active", True)
        .order("seq")
        .execute()
    )
    return res.data or []

def _find_by_seq(chunks: List[Dict[str, Any]], seq: int):
    for c in chunks:
        if int(c.get("seq") or -9999) == int(seq):
            return c
    return None

def _find_next(chunks: List[Dict[str, Any]], seq: int):
    for c in chunks:
        if int(c.get("seq") or 0) > int(seq):
            return c
    return None


# ─────────────────────────────────────────────────────────────
# Lifelong Brain
# ─────────────────────────────────────────────────────────────

def _ensure_student_brain_row(student_id: str):
    try:
        existing = (
            supabase.table("student_brain")
            .select("student_id")
            .eq("student_id", student_id)
            .maybe_single()
            .execute()
        ).data
        if not existing:
            supabase.table("student_brain").insert({"student_id": student_id}).execute()
    except Exception:
        pass

def _get_student_brain_full(student_id: str) -> Dict[str, Any]:
    try:
        res = (
            supabase.table("student_brain")
            .select("*")
            .eq("student_id", student_id)
            .maybe_single()
            .execute()
        )
        return res.data or {}
    except Exception:
        return {}

def _update_student_brain(student_id: str, student_text: str):
    signals = _infer_learning_signals(student_text)
    signals["last_student_text"] = student_text

    brain_state = {
        "difficulty_level": signals["difficulty_level"],
        "confused": signals["confused"],
        "repeat": signals["repeat"],
        "understood": signals["understood"],
        "anxious": signals["anxious"],
        "last_student_text": student_text[:240],
        "updated_at_ms": int(time.time() * 1000),
    }

    payload = {
        "student_id": student_id,
        "stress_score": signals["stress_score"],
        "confidence_score": signals["confidence_score"],
        "brain_state": brain_state,
    }

    try:
        supabase.table("student_brain").upsert(
            payload,
            on_conflict="student_id",
        ).execute()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Progress (Per Chapter)
# ─────────────────────────────────────────────────────────────

def _get_progress(student_id: str, chapter_id: str) -> int:
    try:
        res = (
            supabase.table("student_chapter_progress")
            .select("chunk_seq")
            .eq("student_id", student_id)
            .eq("chapter_id", chapter_id)
            .maybe_single()
            .execute()
        )
        return int(res.data.get("chunk_seq")) if res.data else -1
    except Exception:
        return -1

def _set_progress(student_id: str, chapter_id: str, seq: int):
    try:
        supabase.table("student_chapter_progress").upsert(
            {
                "student_id": student_id,
                "chapter_id": chapter_id,
                "chunk_seq": seq,
            },
            on_conflict="student_id,chapter_id",
        ).execute()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# Learning Signals
# ─────────────────────────────────────────────────────────────

def _clamp(n, lo=0, hi=100):
    return max(lo, min(hi, int(n)))

def _infer_learning_signals(text: str) -> Dict[str, Any]:
    t = (text or "").lower()

    confused = "confused" in t or "don't understand" in t
    repeat = "repeat" in t or "again" in t
    understood = "understood" in t or "got it" in t
    anxious = "scared" in t or "worried" in t

    level = "hard" if confused or anxious else "easy" if understood else "normal"

    stress = 50 + (20 if anxious else 0) + (10 if confused else 0)
    confidence = 50 + (15 if understood else 0) - (10 if confused else 0)

    return {
        "difficulty_level": level,
        "confused": confused,
        "repeat": repeat,
        "understood": understood,
        "anxious": anxious,
        "stress_score": _clamp(stress),
        "confidence_score": _clamp(confidence),
    }


# ─────────────────────────────────────────────────────────────
# Teaching Engine
# ─────────────────────────────────────────────────────────────

def _teach(text: str, title: str):
    return f"{title}\n\n{text}\n\nPause. Tell me what you understood."


def _format_chunk(chunk, teacher):
    return {
        "type": chunk.get("type") or "chunk",
        "seq": chunk.get("seq"),
        "title": chunk.get("title"),
        "text": _teach(chunk.get("chunk_text") or "", chunk.get("title") or ""),
        "media_url": chunk.get("media_url"),
        "duration_sec": chunk.get("duration_sec"),
        "quiz": chunk.get("quiz"),
        "meta": {
            "teacher_name": teacher.get("teacher_name"),
        },
    }


# ─────────────────────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────────────────────

@app.post("/respond")
def respond(payload: RespondPayload):

    _ensure_student_brain_row(payload.student_id)

    teacher = _get_teacher_profile(payload.chapter_id)
    chunks = _load_chunks(payload.chapter_id)
    if not chunks:
        raise HTTPException(status_code=404, detail="No chunks found")

    current_seq = _get_progress(payload.student_id, payload.chapter_id)

    # START
    if payload.action == "start_class":
        first = chunks[0]
        _set_progress(payload.student_id, payload.chapter_id, first["seq"])
        result = _format_chunk(first, teacher)
        result["student_brain"] = _get_student_brain_full(payload.student_id)
        return result

    # NEXT
    if payload.action == "next":
        nxt = _find_next(chunks, current_seq)
        if not nxt:
            return {
                "type": "end",
                "message": "Chapter completed ✅",
                "student_brain": _get_student_brain_full(payload.student_id),
            }

        _set_progress(payload.student_id, payload.chapter_id, nxt["seq"])
        result = _format_chunk(nxt, teacher)
        result["student_brain"] = _get_student_brain_full(payload.student_id)
        return result

    # RESPOND
    if payload.action == "respond":
        if not payload.student_input:
            raise HTTPException(status_code=400, detail="student_input required")

        _update_student_brain(payload.student_id, payload.student_input)

        brain = _get_student_brain_full(payload.student_id)

        return {
            "type": "teacher_reply",
            "reply": "Thank you. Let me explain gently.",
            "student_brain": brain,
        }

    # QUIZ
    if payload.action == "answer_quiz":
        return {
            "type": "quiz_result",
            "correct": True,
            "student_brain": _get_student_brain_full(payload.student_id),
        }
