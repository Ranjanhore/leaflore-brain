# main.py
import os
from typing import Optional, Any, Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client

app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY")
    or os.getenv("VITE_SUPABASE_ANON_KEY")
)

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL / SUPABASE_KEY env vars")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# -----------------------------
# Request / Response Models
# -----------------------------
class RespondPayload(BaseModel):
    action: str  # "start_class" | "respond" | "next" | "answer_quiz"
    student_id: str
    session_id: str
    # IMPORTANT: pass chapter_id from React
    chapter_id: str

    # optional
    student_input: Optional[str] = None
    quiz_answer: Optional[str] = None


# -----------------------------
# Helpers
# -----------------------------
def _load_chunks(chapter_id: str) -> List[Dict[str, Any]]:
    # Uses your schema: public.chapter_chunks with columns:
    # chapter_id, seq, type, title, text, media_url, duration_sec, quiz, is_active
    res = (
        supabase.table("chapter_chunks")
        .select("id,chapter_id,seq,type,title,text,media_url,duration_sec,quiz,is_active")
        .eq("chapter_id", chapter_id)
        .eq("is_active", True)
        .order("seq")
        .execute()
    )
    data = res.data or []
    return data


def _get_state_key(student_id: str, session_id: str, chapter_id: str) -> str:
    # single unique key stored in student_brain
    return f"{student_id}:{session_id}:{chapter_id}"


def _get_progress(state_key: str) -> Dict[str, Any]:
    # student_brain assumed to exist (you already have it in Supabase UI)
    # columns (recommended): state_key (text unique), chunk_seq (int), updated_at
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


def _format_chunk(c: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": c.get("type") or "chunk",
        "seq": c.get("seq"),
        "title": c.get("title"),
        "text": c.get("text"),
        "media_url": c.get("media_url"),
        "duration_sec": c.get("duration_sec"),
        "quiz": c.get("quiz"),  # jsonb
    }


def _find_next(chunks: List[Dict[str, Any]], current_seq: int) -> Optional[Dict[str, Any]]:
    for c in chunks:
        if (c.get("seq") or 0) > current_seq:
            return c
    return None


# -----------------------------
# Core endpoint
# -----------------------------
@app.post("/respond")
def respond(payload: RespondPayload):
    if payload.action not in ("start_class", "respond", "next", "answer_quiz"):
        raise HTTPException(status_code=400, detail="Invalid action")

    chunks = _load_chunks(payload.chapter_id)
    if not chunks:
        return {"type": "error", "message": "No chunks found for this chapter_id."}

    state_key = _get_state_key(payload.student_id, payload.session_id, payload.chapter_id)
    progress = _get_progress(state_key)
    current_seq = int(progress.get("chunk_seq") or -1)

    # --- START CLASS ---
    # Priority:
    # 1) intro chunk (type="intro") if not consumed
    # 2) teacher_greeting chunk if not consumed
    # 3) first normal chunk
    if payload.action == "start_class":
        # find earliest intro
        intro = next((c for c in chunks if (c.get("type") == "intro")), None)
        greeting = next((c for c in chunks if (c.get("type") == "teacher_greeting")), None)

        # if no progress yet -> serve intro (if exists), else greeting, else seq=0
        if current_seq < 0 and intro:
            _set_progress(state_key, int(intro["seq"]))
            return _format_chunk(intro)

        # if intro already served but greeting not served -> serve greeting
        if greeting and current_seq < int(greeting["seq"]):
            _set_progress(state_key, int(greeting["seq"]))
            return _format_chunk(greeting)

        # otherwise serve first normal learning chunk
        first_learning = next((c for c in chunks if (c.get("type") in (None, "", "chunk", "teaching"))), None)
        if first_learning:
            _set_progress(state_key, int(first_learning["seq"]))
            return _format_chunk(first_learning)

        # fallback: first item
        _set_progress(state_key, int(chunks[0]["seq"]))
        return _format_chunk(chunks[0])

    # --- NEXT CHUNK ---
    if payload.action == "next":
        nxt = _find_next(chunks, current_seq)
        if not nxt:
            return {"type": "end", "message": "Chapter completed ✅"}
        _set_progress(state_key, int(nxt["seq"]))
        return _format_chunk(nxt)

    # --- ANSWER QUIZ ---
    if payload.action == "answer_quiz":
        # Check current chunk quiz
        cur = next((c for c in chunks if int(c.get("seq") or -999) == current_seq), None)
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

    # --- RESPOND (teacher reply) ---
    # Here you can:
    # 1) call OpenAI/LLM (your existing leaflore-brain logic)
    # 2) OR keep simple teacher reply and continue sequencing
    student_text = (payload.student_input or "").strip()
    if not student_text:
        return {"type": "error", "message": "student_input required for respond"}

    # Minimal safe reply (replace with your LLM call)
    reply = f"Okay! You said: {student_text}. Now let’s continue."

    # Optionally auto-advance after reply:
    # nxt = _find_next(chunks, current_seq)
    # if nxt:
    #     _set_progress(state_key, int(nxt["seq"]))
    #     return {"type":"teacher_reply","reply":reply,"next":_format_chunk(nxt)}

    return {"type": "teacher_reply", "reply": reply}