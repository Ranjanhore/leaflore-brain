# main.py
import os
import re
from typing import Optional, Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# ───────────────────────────────────────────────────────────────────────────────
# App
# ───────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Leaflore Brain", version="2.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
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
    action: ActionType
    student_id: str
    session_id: str
    chapter_id: str
    student_input: Optional[str] = None
    quiz_answer: Optional[str] = None

# ───────────────────────────────────────────────────────────────────────────────
# Universal Teacher Profile (defaults + optional per-chapter overrides)
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
    "pace": "slow",
    "tone": "gentle",
    "depth": "high",
    "max_paragraph_words": 65,
}

def _load_chapter_meta(chapter_id: str) -> Dict[str, Any]:
    """
    Best-effort metadata loader. If your `chapters` table/columns differ,
    this safely falls back to defaults.
    """
    try:
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
    meta = _load_chapter_meta(chapter_id)
    profile = dict(DEFAULT_TEACHER)

    tp = meta.get("teacher_profile")
    if isinstance(tp, dict):
        profile.update(tp)

    if meta.get("teacher_name"):
        profile["teacher_name"] = meta["teacher_name"]
    if meta.get("voice_id"):
        profile["voice_id"] = meta["voice_id"]

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
# Chunk Helpers (YOUR schema)
# public.chapter_chunks columns (as discussed):
# chapter_id, seq, type, title, chunk_text, media_url, duration_sec, quiz, is_active
# ───────────────────────────────────────────────────────────────────────────────

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

def _get_state_key(student_id: str, session_id: str, chapter_id: str) -> str:
    return f"{student_id}:{session_id}:{chapter_id}"

def _get_progress(state_key: str) -> Dict[str, Any]:
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

# ───────────────────────────────────────────────────────────────────────────────
# Universal Teaching Engine (slow + child-friendly + strong neuro-psych)
# ───────────────────────────────────────────────────────────────────────────────

def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _chunk_sentences(text: str) -> List[str]:
    t = (text or "").replace("\n", " ").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+", t)
    return [p.strip() for p in parts if p.strip()]

def _soften_language(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    s = re.sub(r"^(No[,!\s]+)", "Not exactly, and that's okay. ", s, flags=re.IGNORECASE)
    s = re.sub(r"^(Wrong[,!\s]+)", "Almost. Let’s fix it gently. ", s, flags=re.IGNORECASE)
    return s

def _limit_paragraph_words(text: str, max_words: int) -> List[str]:
    words = (text or "").split()
    if len(words) <= max_words:
        return [text.strip()]
    blocks: List[str] = []
    cur: List[str] = []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words:
            blocks.append(" ".join(cur).strip())
            cur = []
    if cur:
        blocks.append(" ".join(cur).strip())
    return blocks

def _teach_slowly(raw_text: str, title: str, teacher: Dict[str, Any], chunk_type: str) -> str:
    tname = teacher.get("teacher_name", "Teacher")
    grade = teacher.get("grade", "")
    subject = teacher.get("subject", "")
    chapter_name = teacher.get("chapter_name", "")

    raw_text = _clean_spaces(raw_text)
    title = _clean_spaces(title)

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

    if not raw_text:
        return (
            "Okay my dear. Let’s go slowly.\n\n"
            "Look at the visuals for a moment. Then tell me one thing you notice."
        ).strip()

    sentences = [_soften_language(s) for s in _chunk_sentences(raw_text)]
    script_parts: List[str] = []

    if title:
        script_parts.append(f"Okay. Now we will learn: {title}.")
    else:
        script_parts.append("Okay. Now we will learn one important part, slowly.")

    script_parts.append("Please listen calmly. I will speak in small steps.")
    script_parts.append("Let’s break this into tiny pieces.")

    max_words = int(teacher.get("max_paragraph_words") or 65)
    joined = " ".join(sentences).strip()

    for block in _limit_paragraph_words(joined, max_words=max_words):
        script_parts.append(block)
        script_parts.append("Pause. Are you with me till here?")
        script_parts.append("If you want, say: repeat slowly.")

    script_parts.append("Now, a tiny example to make it easy.")
    script_parts.append("Imagine you are explaining this to a younger friend in one line.")

    script_parts.append("Quick check question.")
    script_parts.append("Tell me: what is the main point you understood? Just one sentence.")

    return "\n\n".join([p.strip() for p in script_parts if p.strip()]).strip()

def _teacher_reply(student_text: str, current_chunk: Optional[Dict[str, Any]], teacher: Dict[str, Any]) -> str:
    student_text = _clean_spaces(student_text)
    if not student_text:
        return "I’m here. Tell me your question slowly, in simple words."

    lowered = student_text.lower()

    if any(k in lowered for k in ["repeat", "again", "one more time", "slow", "slowly"]):
        if current_chunk:
            raw = (current_chunk.get("chunk_text") or current_chunk.get("text") or "").strip()
            if raw:
                return (
                    "Of course, my dear.\n\n"
                    "I will repeat slowly.\n\n"
                    f"{_teach_slowly(raw, (current_chunk.get('title') or ''), teacher, 'chunk')}"
                ).strip()
        return "Of course, my dear. Tell me which line you want me to repeat."

    if any(k in lowered for k in ["i don't understand", "dont understand", "confused", "not clear", "hard"]):
        return (
            "It’s completely okay.\n\n"
            "Your brain is learning — confusion is a normal part of learning.\n\n"
            "Tell me which word feels confusing. I will explain that one word first."
        ).strip()

    cur_title = _clean_spaces((current_chunk or {}).get("title") or "")
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

def _format_chunk(c: Dict[str, Any], teacher_profile: Dict[str, Any]) -> Dict[str, Any]:
    chunk_type = (c.get("type") or "chunk")
    if isinstance(chunk_type, str):
        chunk_type = chunk_type.strip() or "chunk"

    raw_title = (c.get("title") or "").strip()
    # ✅ FIX: your DB uses chunk_text (fallback to text if present)
    raw_text = (c.get("chunk_text") or c.get("text") or "").strip()

    if chunk_type in ("chunk", "teaching", "teacher_greeting") or chunk_type is None or chunk_type == "":
        enriched_text = _teach_slowly(
            raw_text=raw_text,
            title=raw_title,
            teacher=teacher_profile,
            chunk_type=str(chunk_type or "chunk"),
        )
    else:
        enriched_text = raw_text

    return {
        "type": chunk_type or "chunk",
        "seq": c.get("seq"),
        "title": raw_title,
        "text": enriched_text,
        "media_url": c.get("media_url"),
        "duration_sec": c.get("duration_sec"),
        "quiz": c.get("quiz"),
        "meta": {
            "teacher_name": teacher_profile.get("teacher_name"),
            "voice_id": teacher_profile.get("voice_id", "default"),
        },
    }

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
    if payload.action == "start_class":
        intro = next((c for c in chunks if (c.get("type") == "intro")), None)
        greeting = next((c for c in chunks if (c.get("type") == "teacher_greeting")), None)

        if current_seq < 0 and intro:
            _set_progress(state_key, int(intro["seq"]))
            return _format_chunk(intro, teacher)

        if greeting and current_seq < int(greeting["seq"]):
            _set_progress(state_key, int(greeting["seq"]))
            return _format_chunk(greeting, teacher)

        first_learning = next((c for c in chunks if (c.get("type") in (None, "", "chunk", "teaching"))), None)
        if first_learning:
            _set_progress(state_key, int(first_learning["seq"]))
            return _format_chunk(first_learning, teacher)

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

    current_chunk = _find_by_seq(chunks, current_seq)
    reply = _teacher_reply(student_text, current_chunk, teacher)

    return {"type": "teacher_reply", "reply": reply, "meta": {"teacher_name": teacher.get("teacher_name")}}