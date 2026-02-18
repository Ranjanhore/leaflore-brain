# main.py
import os
import re
from typing import Optional, Any, Dict, List, Literal, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# ───────────────────────────────────────────────────────────────────────────────
# App
# ───────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Leaflore Brain", version="2.0.0")

# CORS (adjust allow_origins in production if you want)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

# ───────────────────────────────────────────────────────────────────────────────
# Supabase
# ───────────────────────────────────────────────────────────────────────────────

SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or os.getenv("VITE_SUPABASE_ANON_KEY")
)

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL / SUPABASE_KEY env vars")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ───────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ───────────────────────────────────────────────────────────────────────────────

ActionType = Literal["start_class", "respond", "next", "answer_quiz"]

class RespondPayload(BaseModel):
    action: ActionType  # "start_class" | "respond" | "next" | "answer_quiz"
    student_id: str
    session_id: str
    chapter_id: str  # IMPORTANT: pass chapter_id from React

    # optional
    student_input: Optional[str] = None
    quiz_answer: Optional[str] = None


# ───────────────────────────────────────────────────────────────────────────────
# Universal Teacher Profile (per chapter/subject possible)
# ───────────────────────────────────────────────────────────────────────────────

DEFAULT_TEACHER: Dict[str, Any] = {
    "teacher_name": "Anaya Ma'am",
    "persona": (
        "You are a very soft-spoken, warm, friendly teacher. "
        "You have a PhD in Pediatric Neuro (neurodevelopment + child psychology). "
        "You explain slowly and kindly. "
        "You never shame the student. "
        "You notice confusion early and adjust. "
        "You teach in very small steps, with examples and quick checks."
    ),
    "language": "en-IN",
    "pace": "slow",          # slow / normal
    "tone": "gentle",        # gentle / energetic
    "depth": "high",         # low / medium / high
    "max_paragraph_words": 65,  # keep each speaking block short
}

def _load_chapter_meta(chapter_id: str) -> Dict[str, Any]:
    """
    Best-effort: read metadata from public.chapters (if you have it).
    This does NOT break if the table/columns are missing — it just falls back.
    """
    try:
        # Try common columns; if some don't exist, Supabase will error → catch and fallback
        res = (
            supabase.table("chapters")
            .select("id,chapter_name,board,grade,subject,teacher_profile,teacher_name,voice_id")
            .eq("id", chapter_id)
            .maybe_single()
            .execute()
        )
        return res.data or {}
    except Exception:
        return {}

def _get_teacher_profile(chapter_id: str) -> Dict[str, Any]:
    """
    Universal brain: defaults + optional overrides from chapter row:
      - teacher_profile (jsonb)
      - teacher_name
      - voice_id (stored for your UI, not used here)
    """
    meta = _load_chapter_meta(chapter_id)
    profile = dict(DEFAULT_TEACHER)

    # merge teacher_profile if present
    tp = meta.get("teacher_profile")
    if isinstance(tp, dict):
        profile.update(tp)

    # direct overrides (common fields)
    if meta.get("teacher_name"):
        profile["teacher_name"] = meta["teacher_name"]
    if meta.get("voice_id"):
        profile["voice_id"] = meta["voice_id"]  # optional, for frontend usage

    # keep useful context for greeting
    if meta.get("board"):
        profile["board"] = meta["board"]
    if meta.get("grade") is not None:
        profile["grade"] = meta["grade"]
    if meta.get("subject"):
        profile["subject"] = meta["subject"]
    if meta.get("chapter_name"):
        profile["chapter_name"] = meta["chapter_name"]

    return profile


# ───────────────────────────────────────────────────────────────────────────────
# Chunk Helpers (your schema)
# public.chapter_chunks columns:
# chapter_id, seq, type, title, text, media_url, duration_sec, quiz, is_active
# ───────────────────────────────────────────────────────────────────────────────

def _load_chunks(chapter_id: str) -> List[Dict[str, Any]]:
    res = (
        supabase.table("chapter_chunks")
        .select("id,chapter_id,seq,type,title,text,media_url,duration_sec,quiz,is_active")
        .eq("chapter_id", chapter_id)
        .eq("is_active", True)
        .order("seq")
        .execute()
    )
    return res.data or []

def _get_state_key(student_id: str, session_id: str, chapter_id: str) -> str:
    return f"{student_id}:{session_id}:{chapter_id}"

def _get_progress(state_key: str) -> Dict[str, Any]:
    # expects student_brain(state_key unique, chunk_seq int)
    res = (
        supabase.table("student_brain")
        .select("state_key,chunk_seq")
        .eq("state_key", state_key)
        .maybe_single()
        .execute()
    )
    return res.data or {}

def _set_progress(state_key: str, chunk_seq: int) -> None:
    existing = _get_progress(state_key)
    if existing and existing.get("state_key"):
        supabase.table("student_brain").update({"chunk_seq": chunk_seq}).eq("state_key", state_key).execute()
    else:
        supabase.table("student_brain").insert({"state_key": state_key, "chunk_seq": chunk_seq}).execute()

def _find_by_seq(chunks: List[Dict[str, Any]], seq: int) -> Optional[Dict[str, Any]]:
    for c in chunks:
        if int(c.get("seq") or -999999) == int(seq):
            return c
    return None

def _find_next(chunks: List[Dict[str, Any]], current_seq: int) -> Optional[Dict[str, Any]]:
    for c in chunks:
        if int(c.get("seq") or 0) > int(current_seq):
            return c
    return None

def _format_chunk(c: Dict[str, Any], teacher_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns the exact chunk schema your frontend expects,
    but with "text" transformed into a slow, detailed teaching script.
    """
    chunk_type = (c.get("type") or "chunk").strip() if isinstance(c.get("type"), str) else (c.get("type") or "chunk")
    raw_title = (c.get("title") or "").strip()
    raw_text = (c.get("text") or "").strip()

    # Build the teaching script (only for teaching-ish chunks and greeting-ish chunks)
    if chunk_type in ("chunk", "teaching", "teacher_greeting") or chunk_type is None or chunk_type == "":
        enriched_text = _teach_slowly(
            raw_text=raw_text,
            title=raw_title,
            teacher=teacher_profile,
            chunk_type=chunk_type or "chunk",
        )
    else:
        # For intro/video-only chunks: keep text minimal (don’t talk over intro)
        enriched_text = raw_text

    out = {
        "type": chunk_type or "chunk",
        "seq": c.get("seq"),
        "title": raw_title,
        "text": enriched_text,
        "media_url": c.get("media_url"),
        "duration_sec": c.get("duration_sec"),
        "quiz": c.get("quiz"),  # jsonb
        # Helpful meta (non-breaking)
        "meta": {
            "teacher_name": teacher_profile.get("teacher_name"),
            "voice_id": teacher_profile.get("voice_id", "default"),
        },
    }
    return out


# ───────────────────────────────────────────────────────────────────────────────
# Universal Teaching Engine (slow + child-friendly + strong neuro-psych)
# ───────────────────────────────────────────────────────────────────────────────

def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _chunk_sentences(text: str) -> List[str]:
    t = (text or "").replace("\n", " ").strip()
    if not t:
        return []
    # Light sentence split
    parts = re.split(r"(?<=[\.\!\?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def _soften_language(s: str) -> str:
    # Gentle tone helpers
    s = s.strip()
    if not s:
        return s
    # Avoid harsh openings
    s = re.sub(r"^(No[,!\s]+)", "Not exactly, and that's okay. ", s, flags=re.IGNORECASE)
    s = re.sub(r"^(Wrong[,!\s]+)", "Almost. Let’s fix it gently. ", s, flags=re.IGNORECASE)
    return s

def _limit_paragraph_words(text: str, max_words: int) -> List[str]:
    """
    Split into small speaking blocks so TTS sounds slow and calm.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]

    blocks = []
    cur = []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words:
            blocks.append(" ".join(cur).strip())
            cur = []
    if cur:
        blocks.append(" ".join(cur).strip())
    return blocks

def _teach_slowly(raw_text: str, title: str, teacher: Dict[str, Any], chunk_type: str) -> str:
    """
    Convert chunk text into a slow, detailed, child-friendly micro-lesson.
    Deterministic (no external LLM), so it always works reliably.
    """
    tname = teacher.get("teacher_name", "Teacher")
    board = teacher.get("board", "")
    grade = teacher.get("grade", "")
    subject = teacher.get("subject", "")
    chapter_name = teacher.get("chapter_name", "")

    raw_text = _clean_spaces(raw_text)
    title = _clean_spaces(title)

    # If chunk is greeting type: force friendly intro + neuro-psych strength
    if chunk_type == "teacher_greeting":
        base = (
            f"Hello my dear. I’m {tname}. "
            f"I teach very softly and slowly, step by step. "
            f"I also understand child psychology and how your brain learns best. "
            f"So you can speak freely — no fear, okay?\n\n"
        )
        if subject or chapter_name:
            base += f"Today we are doing {subject or 'your subject'}"
            if grade:
                base += f" for Class {grade}"
            if chapter_name:
                base += f", chapter: {chapter_name}"
            base += ".\n\n"
        if raw_text:
            base += _soften_language(raw_text) + "\n\n"
        base += "First, tell me your name. And how do you feel right now — excited, nervous, or sleepy?"
        return base.strip()

    # Normal teaching chunks
    if not raw_text:
        # still provide a calm transition
        return (
            f"Okay my dear. Let’s go slowly.\n\n"
            f"Look at the visuals for a moment. Then tell me one thing you notice."
        ).strip()

    sentences = _chunk_sentences(raw_text)
    sentences = [_soften_language(s) for s in sentences]

    # Build a structured, slow lesson script
    script_parts: List[str] = []

    # Warm opener
    if title:
        script_parts.append(f"Okay. Now we will learn: {title}.")
    else:
        script_parts.append("Okay. Now we will learn one important part, slowly.")

    script_parts.append("Please listen calmly. I will speak in small steps.")

    # Explain (small blocks)
    # We keep content but add gentle scaffolding
    if len(sentences) <= 2:
        script_parts.append("Here is the idea in very simple words.")
    else:
        script_parts.append("Let’s break this into tiny pieces.")

    # Add the original content, but in small speaking blocks
    max_words = int(teacher.get("max_paragraph_words") or 65)
    joined = " ".join(sentences).strip()
    for block in _limit_paragraph_words(joined, max_words=max_words):
        script_parts.append(block)

        # Neuro-psych supportive check-in after each block
        script_parts.append("Pause. Are you with me till here?")
        script_parts.append("If you want, say: repeat slowly.")

    # Gentle example prompt (universal, doesn’t assume topic)
    script_parts.append("Now, a tiny example to make it easy.")
    script_parts.append("Imagine you are explaining this to a younger friend in one line.")

    # Micro-check question
    script_parts.append("Quick check question.")
    script_parts.append("Tell me: what is the main point you understood? Just one sentence.")

    return "\n\n".join([p.strip() for p in script_parts if p.strip()]).strip()


def _teacher_reply(student_text: str, current_chunk: Optional[Dict[str, Any]], teacher: Dict[str, Any]) -> str:
    """
    Universal, gentle reply that:
    - acknowledges emotions
    - answers briefly (without hallucinating)
    - encourages asking + continuing
    """
    tname = teacher.get("teacher_name", "Teacher")
    student_text = _clean_spaces(student_text)

    # If empty safety
    if not student_text:
        return "I’m here. Tell me your question slowly, in simple words."

    # If student asks to repeat
    lowered = student_text.lower()
    if any(k in lowered for k in ["repeat", "again", "one more time", "slow", "slowly"]):
        if current_chunk and (current_chunk.get("text") or "").strip():
            return (
                f"Of course, my dear.\n\n"
                f"I will repeat slowly.\n\n"
                f"{_teach_slowly(current_chunk.get('text') or '', current_chunk.get('title') or '', teacher, 'chunk')}"
            ).strip()
        return "Of course, my dear. Tell me which line you want me to repeat."

    # If student says they don’t understand
    if any(k in lowered for k in ["i don't understand", "dont understand", "confused", "not clear", "hard"]):
        return (
            f"It’s completely okay.\n\n"
            f"Your brain is learning — confusion is a normal part of learning.\n\n"
            f"Tell me which word feels confusing. I will explain that one word first."
        ).strip()

    # Default: calm response that keeps lesson moving
    cur_title = (current_chunk or {}).get("title") or ""
    cur_title = _clean_spaces(cur_title)

    reply = [
        f"Thank you for telling me.\n\nYou said: “{student_text}”.",
        "I’m listening carefully.",
        "Now I will answer gently.",
        "",
        "First, tell me one detail: are you asking about the meaning, the example, or the diagram/video?",
    ]
    if cur_title:
        reply.append(f"Also, we are currently on: {cur_title}.")

    reply.append("If you want to continue the lesson, you can press Next when you are ready.")
    return "\n".join([r for r in reply if r is not None]).strip()


# ───────────────────────────────────────────────────────────────────────────────
# Core endpoint
# ───────────────────────────────────────────────────────────────────────────────

@app.post("/respond")
def respond(payload: RespondPayload):
    if payload.action not in ("start_class", "respond", "next", "answer_quiz"):
        raise HTTPException(status_code=400, detail="Invalid action")

    teacher = _get_teacher_profile(payload.chapter_id)
    chunks = _load_chunks(payload.chapter_id)
    if not chunks:
        return {"type": "error", "message": "No chunks found for this chapter_id."}

    state_key = _get_state_key(payload.student_id, payload.session_id, payload.chapter_id)
    progress = _get_progress(state_key)
    current_seq = int(progress.get("chunk_seq") or -1)

    # ── START CLASS ──────────────────────────────────────────────────────────
    # Priority:
    # 1) intro chunk (type="intro") if not consumed
    # 2) teacher_greeting chunk if not consumed
    # 3) first normal chunk
    if payload.action == "start_class":
        intro = next((c for c in chunks if (c.get("type") == "intro")), None)
        greeting = next((c for c in chunks if (c.get("type") == "teacher_greeting")), None)

        # If no progress yet -> serve intro (if exists)
        if current_seq < 0 and intro:
            _set_progress(state_key, int(intro["seq"]))
            # intro should not be overwritten with long talk
            return _format_chunk(intro, teacher)

        # If intro served but greeting not served -> serve greeting
        if greeting and current_seq < int(greeting["seq"]):
            _set_progress(state_key, int(greeting["seq"]))
            return _format_chunk(greeting, teacher)

        # Otherwise serve first learning chunk
        first_learning = next((c for c in chunks if (c.get("type") in (None, "", "chunk", "teaching"))), None)
        if first_learning:
            _set_progress(state_key, int(first_learning["seq"]))
            return _format_chunk(first_learning, teacher)

        # Fallback
        _set_progress(state_key, int(chunks[0]["seq"]))
        return _format_chunk(chunks[0], teacher)

    # ── NEXT CHUNK ───────────────────────────────────────────────────────────
    if payload.action == "next":
        nxt = _find_next(chunks, current_seq)
        if not nxt:
            return {"type": "end", "message": "Chapter completed ✅"}
        _set_progress(state_key, int(nxt["seq"]))
        return _format_chunk(nxt, teacher)

    # ── ANSWER QUIZ ──────────────────────────────────────────────────────────
    if payload.action == "answer_quiz":
        cur = _find_by_seq(chunks, current_seq)
        if not cur:
            return {"type": "error", "message": "No active chunk to answer."}

        quiz = cur.get("quiz") or {}
        expected = (quiz.get("answer") or "").strip().lower()
        given = (payload.quiz_answer or "").strip().lower()

        ok = expected != "" and given == expected
        return {
            "type": "quiz_result",
            "correct": ok,
            "expected": quiz.get("answer"),
            "explanation": quiz.get("explanation") or "",
        }

    # ── RESPOND (teacher reply) ──────────────────────────────────────────────
    student_text = (payload.student_input or "").strip()
    if not student_text:
        return {"type": "error", "message": "student_input required for respond"}

    # Current chunk context (so teacher can say “we are on …”)
    current_chunk = _find_by_seq(chunks, current_seq)

    # Soft, universal teacher reply
    reply = _teacher_reply(student_text, current_chunk, teacher)

    # OPTIONAL: If you want auto-advance after reply, uncomment this block.
    # nxt = _find_next(chunks, current_seq)
    # if nxt:
    #     _set_progress(state_key, int(nxt["seq"]))
    #     return {"type": "teacher_reply", "reply": reply, "next": _format_chunk(nxt, teacher)}

    return {"type": "teacher_reply", "reply": reply, "meta": {"teacher_name": teacher.get("teacher_name")}}
