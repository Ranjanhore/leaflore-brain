# main.py
import os
import re
import time
from typing import Optional, Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# ───────────────────────────────────────────────────────────────────────────────
# App
# ───────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Leaflore Brain", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production if needed
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
    session_id: str  # kept for telemetry/future, NOT used for lifelong progress
    chapter_id: str

    student_input: Optional[str] = None
    quiz_answer: Optional[str] = None


# ───────────────────────────────────────────────────────────────────────────────
# Universal Teacher Profile (override-able per chapter)
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
    "max_paragraph_words": 65,  # base (will adapt)
}


def _load_chapter_meta(chapter_id: str) -> Dict[str, Any]:
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
# Chunk Helpers
# ───────────────────────────────────────────────────────────────────────────────

def _load_chunks(chapter_id: str) -> List[Dict[str, Any]]:
    """
    Supports your current schema where the column is `chunk_text`.
    """
    res = (
        supabase.table("chapter_chunks")
        .select("id,chapter_id,seq,type,title,chunk_text,media_url,duration_sec,quiz,is_active")
        .eq("chapter_id", chapter_id)
        .eq("is_active", True)
        .order("seq")
        .execute()
    )
    return res.data or []


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
# Lifelong Brain + Per-chapter Progress (MODE C)
#   - student_brain: lifelong memory (one row per student_id) + optional state_key
#   - student_chapter_progress: progress per (student_id, chapter_id)
# ───────────────────────────────────────────────────────────────────────────────

def _ensure_student_brain_row(student_id: str) -> None:
    """
    Best-effort: ensure a lifelong row exists for this student.
    If your student_brain also has `state_key`, we set it = student_id (safe).
    """
    try:
        existing = (
            supabase.table("student_brain")
            .select("id,student_id")
            .eq("student_id", student_id)
            .maybe_single()
            .execute()
        ).data

        if existing and existing.get("student_id"):
            return

        # Insert minimal row; if state_key column exists, this still works
        # (Supabase ignores unknown keys? It will error if column missing; so we keep it minimal)
        supabase.table("student_brain").insert({"student_id": student_id}).execute()
    except Exception:
        return


def _get_progress(student_id: str, chapter_id: str) -> int:
    """
    Reads from student_chapter_progress (lifelong progress).
    Falls back to -1 if no row or table missing.
    """
    try:
        res = (
            supabase.table("student_chapter_progress")
            .select("student_id,chapter_id,chunk_seq")
            .eq("student_id", student_id)
            .eq("chapter_id", chapter_id)
            .maybe_single()
            .execute()
        )
        data = res.data or {}
        return int(data.get("chunk_seq") if data.get("chunk_seq") is not None else -1)
    except Exception:
        return -1


def _set_progress(student_id: str, chapter_id: str, chunk_seq: int) -> None:
    """
    Upsert per-chapter progress.
    Requires UNIQUE(student_id, chapter_id) on student_chapter_progress.
    """
    try:
        supabase.table("student_chapter_progress").upsert(
            {"student_id": student_id, "chapter_id": chapter_id, "chunk_seq": int(chunk_seq)},
            on_conflict="student_id,chapter_id",
        ).execute()
    except Exception:
        try:
            supabase.table("student_chapter_progress").insert(
                {"student_id": student_id, "chapter_id": chapter_id, "chunk_seq": int(chunk_seq)}
            ).execute()
        except Exception:
            try:
                supabase.table("student_chapter_progress").update(
                    {"chunk_seq": int(chunk_seq)}
                ).eq("student_id", student_id).eq("chapter_id", chapter_id).execute()
            except Exception:
                return


# ───────────────────────────────────────────────────────────────────────────────
# Teaching Engine (deterministic) + B/C Adaptation
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
    s = s.strip()
    if not s:
        return s
    s = re.sub(r"^(No[,!\s]+)", "Not exactly, and that's okay. ", s, flags=re.IGNORECASE)
    s = re.sub(r"^(Wrong[,!\s]+)", "Almost. Let’s fix it gently. ", s, flags=re.IGNORECASE)
    return s


def _limit_paragraph_words(text: str, max_words: int) -> List[str]:
    words = text.split()
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


def _clamp(n: int, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, int(n)))


def _infer_learning_signals(student_text: str) -> Dict[str, Any]:
    t = (student_text or "").lower().strip()

    confused = any(k in t for k in [
        "confused", "not clear", "dont understand", "don't understand", "i don't get", "i dont get",
        "hard", "difficult", "can't understand", "cannot understand"
    ])
    repeat = any(k in t for k in ["repeat", "again", "one more time", "slow", "slowly", "say again"])
    understood = any(k in t for k in ["understood", "got it", "okay", "ok", "yes", "makes sense", "clear now"])
    anxious = any(k in t for k in ["scared", "anxious", "panic", "worried", "i can't", "hate", "frustrated", "angry"])

    if anxious or confused:
        level = "hard"
    elif understood and not confused:
        level = "easy"
    else:
        level = "normal"

    stress = 50
    confidence = 50

    if anxious:
        stress += 25
        confidence -= 15
    if confused:
        stress += 15
        confidence -= 10
    if repeat:
        stress += 8
        confidence -= 5
    if understood:
        stress -= 12
        confidence += 15

    return {
        "difficulty_level": level,
        "confused": confused,
        "repeat": repeat,
        "understood": understood,
        "anxious": anxious,
        "stress_score": _clamp(stress),
        "confidence_score": _clamp(confidence),
    }


def _update_student_state(student_id: str, signals: Dict[str, Any]) -> None:
    """
    Updates lifelong state in student_brain (row per student_id).
    Columns you already have: stress_score, confidence_score, brain_state, updated_at, etc.
    If you also have state_key, we set it to student_id (safe).
    """
    brain_state = {
        "difficulty_level": signals.get("difficulty_level", "normal"),
        "confused": bool(signals.get("confused")),
        "repeat": bool(signals.get("repeat")),
        "understood": bool(signals.get("understood")),
        "anxious": bool(signals.get("anxious")),
        "last_student_text": (signals.get("last_student_text") or "")[:240],
        "updated_at_ms": int(time.time() * 1000),
    }

    payload: Dict[str, Any] = {
        "student_id": student_id,
        "stress_score": int(signals.get("stress_score", 50)),
        "confidence_score": int(signals.get("confidence_score", 50)),
        "brain_state": brain_state,
    }

    try:
        # try to also set state_key if column exists
        payload["state_key"] = student_id
    except Exception:
        pass

    try:
        existing = (
            supabase.table("student_brain")
            .select("id,student_id")
            .eq("student_id", student_id)
            .maybe_single()
            .execute()
        ).data

        if existing and existing.get("student_id"):
            supabase.table("student_brain").update(payload).eq("student_id", student_id).execute()
        else:
            supabase.table("student_brain").insert(payload).execute()
    except Exception:
        # never block class
        return


def _get_student_brain_state(student_id: str) -> Dict[str, Any]:
    try:
        res = (
            supabase.table("student_brain")
            .select("stress_score,confidence_score,brain_state")
            .eq("student_id", student_id)
            .maybe_single()
            .execute()
        )
        return res.data or {}
    except Exception:
        return {}


def _teaching_knobs(teacher: Dict[str, Any], student_state: Dict[str, Any]) -> Dict[str, Any]:
    brain_state = (student_state or {}).get("brain_state") or {}
    level = str(brain_state.get("difficulty_level") or "normal").lower()

    if level == "hard":
        return {"max_words": 40, "extra_checks": 2, "more_examples": True, "level": "hard"}
    if level == "easy":
        return {"max_words": 80, "extra_checks": 1, "more_examples": False, "level": "easy"}
    return {
        "max_words": int(teacher.get("max_paragraph_words") or 65),
        "extra_checks": 1,
        "more_examples": True,
        "level": "normal",
    }


def _teach_slowly(raw_text: str, title: str, teacher: Dict[str, Any], chunk_type: str, student_state: Dict[str, Any]) -> str:
    tname = teacher.get("teacher_name", "Teacher")
    grade = teacher.get("grade", "")
    subject = teacher.get("subject", "")
    chapter_name = teacher.get("chapter_name", "")

    knobs = _teaching_knobs(teacher, student_state)
    max_words = int(knobs["max_words"])
    extra_checks = int(knobs["extra_checks"])
    more_examples = bool(knobs["more_examples"])
    level = str(knobs["level"])

    raw_text = _clean_spaces(raw_text)
    title = _clean_spaces(title)

    if chunk_type == "teacher_greeting":
        # greeting adapts too
        warmth = (
            "If anything feels confusing, it is completely okay. "
            "We will go slowly, and I will repeat as many times as you want.\n\n"
            if level == "hard"
            else ""
        )
        base = (
            f"Hello my dear. I’m {tname}. "
            f"I teach very softly and slowly, step by step. "
            f"I also understand child psychology and how your brain learns best. "
            f"So you can speak freely — no fear, okay?\n\n"
            f"{warmth}"
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

    if level == "hard":
        script_parts.append("We will go very slowly. Small steps. No hurry.")
    else:
        script_parts.append("Please listen calmly. I will speak in small steps.")

    script_parts.append(
        "Let’s break this into tiny pieces." if len(sentences) > 2 else "Here is the idea in very simple words."
    )

    joined = " ".join(sentences).strip()
    for block in _limit_paragraph_words(joined, max_words=max_words):
        script_parts.append(block)
        for _ in range(extra_checks):
            script_parts.append("Pause. Are you with me till here?")
            script_parts.append("If you want, say: repeat slowly.")

    if more_examples:
        script_parts.append("Now, a tiny example to make it easy.")
        script_parts.append("Imagine you are explaining this to a younger friend in one line.")
    else:
        script_parts.append("Quick example in one line: say it in your own words.")

    script_parts.append("Quick check question.")
    script_parts.append("Tell me: what is the main point you understood? Just one sentence.")

    return "\n\n".join([p.strip() for p in script_parts if p.strip()]).strip()


def _teacher_reply(student_text: str, current_chunk: Optional[Dict[str, Any]], teacher: Dict[str, Any], student_state: Dict[str, Any]) -> str:
    student_text = _clean_spaces(student_text)
    if not student_text:
        return "I’m here. Tell me your question slowly, in simple words."

    knobs = _teaching_knobs(teacher, student_state)
    level = str(knobs["level"])

    lowered = student_text.lower()

    if any(k in lowered for k in ["repeat", "again", "one more time", "slow", "slowly"]):
        if current_chunk:
            raw = (current_chunk.get("chunk_text") or current_chunk.get("text") or "").strip()
            if raw:
                return (
                    "Of course, my dear.\n\n"
                    "I will repeat slowly.\n\n"
                    f"{_teach_slowly(raw, current_chunk.get('title') or '', teacher, 'chunk', student_state)}"
                ).strip()
        return "Of course, my dear. Tell me which line you want me to repeat."

    if any(k in lowered for k in ["i don't understand", "dont understand", "confused", "not clear", "hard"]):
        extra = (
            "\n\nTake a small breath. You are safe here. "
            "Tell me only ONE word that feels confusing."
            if level == "hard"
            else "\n\nTell me which word feels confusing. I will explain that one word first."
        )
        return (
            "It’s completely okay.\n\n"
            "Your brain is learning — confusion is a normal part of learning."
            f"{extra}"
        ).strip()

    cur_title = _clean_spaces((current_chunk or {}).get("title") or "")

    if level == "hard":
        opener = "Thank you for telling me.\n\nI’m listening very carefully."
        next_hint = "If you want, ask me to repeat slowly or give one example."
    elif level == "easy":
        opener = "Nice. I got you.\n\nTell me your exact doubt in one line."
        next_hint = "If you’re ready, you can press Next to continue."
    else:
        opener = "Thank you for telling me.\n\nI’m listening carefully."
        next_hint = "If you want to continue the lesson, you can press Next when you are ready."

    reply = [
        opener,
        f'You said: “{student_text}”.',
        "",
        "One quick question so I answer perfectly:",
        "Are you asking about the meaning, the example, or the diagram/video?",
    ]
    if cur_title:
        reply.append(f"Also, we are currently on: {cur_title}.")
    reply.append(next_hint)
    return "\n".join([r for r in reply if r is not None]).strip()


def _format_chunk(c: Dict[str, Any], teacher_profile: Dict[str, Any], student_state: Dict[str, Any]) -> Dict[str, Any]:
    chunk_type = (c.get("type") or "chunk")
    if isinstance(chunk_type, str):
        chunk_type = chunk_type.strip() or "chunk"

    raw_title = (c.get("title") or "").strip()
    raw_text = ((c.get("chunk_text") or c.get("text")) or "").strip()

    if chunk_type in ("chunk", "teaching", "teacher_greeting") or chunk_type is None or chunk_type == "":
        enriched_text = _teach_slowly(
            raw_text=raw_text,
            title=raw_title,
            teacher=teacher_profile,
            chunk_type=chunk_type or "chunk",
            student_state=student_state,
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
            # helpful debug for you (non-breaking)
            "difficulty_level": ((student_state or {}).get("brain_state") or {}).get("difficulty_level", "normal"),
        },
    }


# ───────────────────────────────────────────────────────────────────────────────
# Endpoint
# ───────────────────────────────────────────────────────────────────────────────

@app.post("/respond")
def respond(payload: RespondPayload):
    if payload.action not in ("start_class", "respond", "next", "answer_quiz"):
        raise HTTPException(status_code=400, detail="Invalid action")

    # Ensure lifelong brain row exists (non-blocking)
    _ensure_student_brain_row(payload.student_id)

    teacher = _get_teacher_profile(payload.chapter_id)
    chunks = _load_chunks(payload.chapter_id)
    if not chunks:
        return {"type": "error", "message": "No chunks found for this chapter_id."}

    # Current student brain state (B/C)
    student_state = _get_student_brain_state(payload.student_id)

    # MODE C progress: (student_id, chapter_id)
    current_seq = _get_progress(payload.student_id, payload.chapter_id)

    # ── START CLASS ──────────────────────────────────────────────────────────
    if payload.action == "start_class":
        intro = next((c for c in chunks if (c.get("type") == "intro")), None)
        greeting = next((c for c in chunks if (c.get("type") == "teacher_greeting")), None)

        if current_seq < 0 and intro:
            _set_progress(payload.student_id, payload.chapter_id, int(intro["seq"]))
            return _format_chunk(intro, teacher, student_state)

        if greeting and current_seq < int(greeting["seq"]):
            _set_progress(payload.student_id, payload.chapter_id, int(greeting["seq"]))
            return _format_chunk(greeting, teacher, student_state)

        first_learning = next((c for c in chunks if (c.get("type") in (None, "", "chunk", "teaching"))), None)
        if first_learning:
            _set_progress(payload.student_id, payload.chapter_id, int(first_learning["seq"]))
            return _format_chunk(first_learning, teacher, student_state)

        _set_progress(payload.student_id, payload.chapter_id, int(chunks[0]["seq"]))
        return _format_chunk(chunks[0], teacher, student_state)

    # ── NEXT CHUNK ───────────────────────────────────────────────────────────
    if payload.action == "next":
        nxt = _find_next(chunks, current_seq)
        if not nxt:
            return {"type": "end", "message": "Chapter completed ✅"}
        _set_progress(payload.student_id, payload.chapter_id, int(nxt["seq"]))
        return _format_chunk(nxt, teacher, student_state)

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

    # ── RESPOND (B + C) ──────────────────────────────────────────────────────
    student_text = (payload.student_input or "").strip()
    if not student_text:
        return {"type": "error", "message": "student_input required for respond"}

    # B) infer signals + update lifelong brain
    signals = _infer_learning_signals(student_text)
    signals["last_student_text"] = student_text
    _update_student_state(payload.student_id, signals)

    # Refresh state (so C adapts immediately)
    student_state = _get_student_brain_state(payload.student_id)

    current_chunk = _find_by_seq(chunks, current_seq)
    reply = _teacher_reply(student_text, current_chunk, teacher, student_state)

    return {
        "type": "teacher_reply",
        "reply": reply,
        "meta": {
            "teacher_name": teacher.get("teacher_name"),
            "difficulty_level": ((student_state or {}).get("brain_state") or {}).get("difficulty_level", "normal"),
            "stress_score": (student_state or {}).get("stress_score", 50),
            "confidence_score": (student_state or {}).get("confidence_score", 50),
        },
    }
