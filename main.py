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
app = FastAPI(title="Leaflore Brain", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
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
# Request Model
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
# Universal Teacher Profile (defaults + optional chapter overrides)
# ───────────────────────────────────────────────────────────────────────────────
DEFAULT_TEACHER: Dict[str, Any] = {
    "teacher_name": "Anaya Ma'am",
    "persona": (
        "You are a very soft-spoken, warm, friendly teacher. "
        "You have a PhD in Pediatric Neuro (neurodevelopment + child psychology). "
        "You teach slowly, gently, and in tiny steps. "
        "You never shame the student. You normalize confusion and guide calmly."
    ),
    "language": "en-IN",
    "pace": "slow",
    "tone": "gentle",
    "depth": "high",
    "max_paragraph_words": 60,
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

    # context (nice for greetings)
    for k in ("board", "grade", "subject", "chapter_name"):
        if meta.get(k) is not None:
            profile[k] = meta[k]

    return profile

# ───────────────────────────────────────────────────────────────────────────────
# Chunks
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

# ───────────────────────────────────────────────────────────────────────────────
# Student psychology state (stored in student_brain columns)
# ───────────────────────────────────────────────────────────────────────────────
def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

def _get_progress(state_key: str) -> Dict[str, Any]:
    res = (
        supabase.table("student_brain")
        .select(
            "state_key,chunk_seq,confidence,confusion_count,repeat_count,attention_flags,pace,depth,last_emotion"
        )
        .eq("state_key", state_key)
        .maybe_single()
        .execute()
    )
    return res.data or {}

def _upsert_progress(state_key: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses unique index on state_key.
    """
    payload = {"state_key": state_key, **patch}
    try:
        supabase.table("student_brain").upsert(payload).execute()
    except Exception:
        # fallback if upsert not available in your client version
        existing = _get_progress(state_key)
        if existing.get("state_key"):
            supabase.table("student_brain").update(patch).eq("state_key", state_key).execute()
        else:
            supabase.table("student_brain").insert(payload).execute()
    return _get_progress(state_key)

def _set_chunk_seq(state_key: str, chunk_seq: int) -> None:
    _upsert_progress(state_key, {"chunk_seq": int(chunk_seq), "updated_at": "now()"})

# ───────────────────────────────────────────────────────────────────────────────
# Emotion detection + adaptive teaching knobs
# ───────────────────────────────────────────────────────────────────────────────
CONFUSION_KEYS = [
    "don't understand", "dont understand", "not clear", "confused", "hard", "difficult",
    "what is", "meaning", "i can't", "i cannot", "can't understand", "cannot understand",
]
REPEAT_KEYS = ["repeat", "again", "one more time", "slow", "slowly", "say again"]
BORED_KEYS = ["boring", "tired", "sleepy", "later", "not interested", "too long"]
SHY_KEYS = ["i am not sure", "not sure", "maybe", "i think", "sorry", "i guess"]
EXCITED_KEYS = ["wow", "cool", "amazing", "tell me more", "why", "how", "more", "what if"]

def _clean_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _detect_emotion(student_text: str) -> str:
    t = (student_text or "").lower()
    if any(k in t for k in REPEAT_KEYS):
        return "repeat"
    if any(k in t for k in CONFUSION_KEYS):
        return "confused"
    if any(k in t for k in BORED_KEYS):
        return "bored"
    if any(k in t for k in EXCITED_KEYS):
        return "excited"
    if any(k in t for k in SHY_KEYS):
        return "shy"
    return "neutral"

def _update_student_profile(state_key: str, emotion: str) -> Dict[str, Any]:
    """
    Update counters + confidence, return updated progress.
    """
    cur = _get_progress(state_key)
    confidence = int(cur.get("confidence") or 70)
    confusion_count = int(cur.get("confusion_count") or 0)
    repeat_count = int(cur.get("repeat_count") or 0)
    attention_flags = int(cur.get("attention_flags") or 0)

    if emotion == "confused":
        confidence -= 8
        confusion_count += 1
    elif emotion == "repeat":
        confidence -= 4
        repeat_count += 1
    elif emotion == "bored":
        attention_flags += 1
        confidence -= 2
    elif emotion == "excited":
        confidence += 4
    elif emotion == "shy":
        confidence -= 2

    confidence = _clamp(confidence, 0, 100)

    # gentle auto-adjust: if repeated confusion, force slower + smaller chunks
    pace = cur.get("pace") or "slow"
    depth = cur.get("depth") or "high"
    if confusion_count >= 2 or repeat_count >= 2:
        pace = "slow"
        depth = "high"
    if attention_flags >= 2:
        # reduce cognitive load
        depth = "medium"

    patch = {
        "confidence": confidence,
        "confusion_count": confusion_count,
        "repeat_count": repeat_count,
        "attention_flags": attention_flags,
        "pace": pace,
        "depth": depth,
        "last_emotion": emotion,
        "updated_at": "now()",
    }
    return _upsert_progress(state_key, patch)

def _soften_language(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    s = re.sub(r"^(No[,!\s]+)", "Not exactly, and that’s okay. ", s, flags=re.IGNORECASE)
    s = re.sub(r"^(Wrong[,!\s]+)", "Almost. Let’s fix it gently. ", s, flags=re.IGNORECASE)
    return s

def _limit_paragraph_words(text: str, max_words: int) -> List[str]:
    words = (text or "").split()
    if len(words) <= max_words:
        return [(text or "").strip()]
    blocks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= max_words:
            blocks.append(" ".join(cur).strip())
            cur = []
    if cur:
        blocks.append(" ".join(cur).strip())
    return blocks

def _teach_slowly(raw_text: str, title: str, teacher: Dict[str, Any], student: Dict[str, Any], chunk_type: str) -> str:
    """
    Deterministic slow teaching + adaptive micro-support based on student profile.
    """
    tname = teacher.get("teacher_name", "Teacher")
    raw_text = _clean_spaces(raw_text)
    title = _clean_spaces(title)

    last_emotion = (student.get("last_emotion") or "neutral").lower()
    confidence = int(student.get("confidence") or 70)

    # Adaptive knobs
    base_max = int(teacher.get("max_paragraph_words") or 60)
    if last_emotion in ("confused", "repeat") or confidence <= 50:
        max_words = max(35, min(base_max, 45))
    elif last_emotion == "bored":
        max_words = max(30, min(base_max, 40))
    else:
        max_words = base_max

    reassurance = ""
    if last_emotion in ("confused", "repeat") or confidence <= 50:
        reassurance = (
            "It’s completely okay if this feels confusing. "
            "Your brain is learning, and we will go step by step."
        )
    elif last_emotion == "shy":
        reassurance = "No pressure. Even a small answer is good. I’m right here with you."
    elif last_emotion == "bored":
        reassurance = "Let’s make it quick and interesting. One small point at a time."
    elif last_emotion == "excited":
        reassurance = "I love your curiosity. Let’s understand it nicely."

    if chunk_type == "teacher_greeting":
        msg = [
            f"Hello my dear. I’m {tname}.",
            "I speak softly and slowly.",
            "I also understand child psychology and how your brain learns best.",
            "So you can ask freely — no fear, okay?",
        ]
        if reassurance:
            msg.append(reassurance)
        if raw_text:
            msg.append(_soften_language(raw_text))
        msg.append("First, tell me your name. And how are you feeling right now?")
        return "\n\n".join([m for m in msg if m]).strip()

    if not raw_text:
        return "Okay my dear. Look at the picture/video for a moment. Tell me one thing you notice."

    script: List[str] = []
    if title:
        script.append(f"Okay. Now we will learn: {title}.")
    else:
        script.append("Okay. Now we will learn something important, slowly.")

    if reassurance:
        script.append(reassurance)

    script.append("Please listen calmly. I will speak in tiny steps.")

    blocks = _limit_paragraph_words(_soften_language(raw_text), max_words=max_words)
    for b in blocks:
        script.append(b)
        # micro-checks (short, not annoying)
        if last_emotion in ("confused", "repeat") or confidence <= 50:
            script.append("Pause. Are you with me till here?")
            script.append("If you want, say: repeat slowly.")
        elif last_emotion == "bored":
            script.append("Quick check: can you say the main word you heard?")
        else:
            script.append("Quick check: what did you understand in one line?")

    # end prompt
    script.append("Now tell me your one-sentence understanding.")
    return "\n\n".join([s for s in script if s]).strip()

def _format_chunk(c: Dict[str, Any], teacher: Dict[str, Any], student: Dict[str, Any]) -> Dict[str, Any]:
    chunk_type = (c.get("type") or "chunk") if isinstance(c.get("type"), str) else (c.get("type") or "chunk")
    chunk_type = (chunk_type or "chunk").strip() if isinstance(chunk_type, str) else "chunk"

    title = (c.get("title") or "").strip()
    raw_text = (c.get("chunk_text") or c.get("text") or "").strip()

    if chunk_type in ("chunk", "teaching", "teacher_greeting", "", None):
        text = _teach_slowly(raw_text, title, teacher, student, chunk_type or "chunk")
    else:
        text = raw_text  # intro/video/etc

    return {
        "type": chunk_type or "chunk",
        "seq": c.get("seq"),
        "title": title,
        "text": text,
        "media_url": c.get("media_url"),
        "duration_sec": c.get("duration_sec"),
        "quiz": c.get("quiz"),
        "meta": {
            "teacher_name": teacher.get("teacher_name"),
            "voice_id": teacher.get("voice_id", "default"),
            "student_confidence": int(student.get("confidence") or 70),
            "student_emotion": student.get("last_emotion") or "neutral",
        },
    }

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

def _teacher_reply(student_text: str, current_chunk: Optional[Dict[str, Any]], teacher: Dict[str, Any], student: Dict[str, Any]) -> str:
    tname = teacher.get("teacher_name", "Teacher")
    student_text = _clean_spaces(student_text)
    emotion = (student.get("last_emotion") or "neutral").lower()
    confidence = int(student.get("confidence") or 70)

    if not student_text:
        return "I’m here. Tell me your question slowly, in simple words."

    if emotion in ("repeat", "confused") or confidence <= 50:
        support = "It’s okay. We will go slowly. No pressure."
    elif emotion == "shy":
        support = "No worries. Even a small try is perfect."
    else:
        support = "I’m listening carefully."

    cur_title = ""
    if current_chunk:
        cur_title = (current_chunk.get("title") or "").strip()

    parts = [
        f"Thank you, my dear.",
        support,
        f"You said: “{student_text}”.",
        "Tell me one thing: do you want the meaning, an example, or to repeat the steps?",
    ]
    if cur_title:
        parts.append(f"We are currently on: {cur_title}.")
    parts.append("When you feel ready, press Next to continue.")
    return "\n\n".join([p for p in parts if p]).strip()

# ───────────────────────────────────────────────────────────────────────────────
# Endpoint
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

    # Ensure student profile row exists (on first touch)
    progress = _get_progress(state_key)
    if not progress.get("state_key"):
        progress = _upsert_progress(state_key, {"chunk_seq": -1})

    current_seq = int(progress.get("chunk_seq") or -1)

    # ── START CLASS ──────────────────────────────────────────────────────────
    if payload.action == "start_class":
        # reset-ish but keep psych profile; only ensure seq starts clean if new session
        intro = next((c for c in chunks if (c.get("type") == "intro")), None)
        greeting = next((c for c in chunks if (c.get("type") == "teacher_greeting")), None)

        if current_seq < 0 and intro:
            _set_chunk_seq(state_key, int(intro["seq"]))
            progress = _get_progress(state_key)
            return _format_chunk(intro, teacher, progress)

        if greeting and current_seq < int(greeting["seq"]):
            _set_chunk_seq(state_key, int(greeting["seq"]))
            progress = _get_progress(state_key)
            return _format_chunk(greeting, teacher, progress)

        first_learning = next((c for c in chunks if (c.get("type") in (None, "", "chunk", "teaching"))), None)
        if first_learning:
            _set_chunk_seq(state_key, int(first_learning["seq"]))
            progress = _get_progress(state_key)
            return _format_chunk(first_learning, teacher, progress)

        _set_chunk_seq(state_key, int(chunks[0]["seq"]))
        progress = _get_progress(state_key)
        return _format_chunk(chunks[0], teacher, progress)

    # ── NEXT ────────────────────────────────────────────────────────────────
    if payload.action == "next":
        nxt = _find_next(chunks, current_seq)
        if not nxt:
            return {"type": "end", "message": "Chapter completed ✅"}
        _set_chunk_seq(state_key, int(nxt["seq"]))
        progress = _get_progress(state_key)
        return _format_chunk(nxt, teacher, progress)

    # ── ANSWER QUIZ ─────────────────────────────────────────────────────────
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

    # ── RESPOND (psych-adaptive teacher reply) ───────────────────────────────
    student_text = (payload.student_input or "").strip()
    if not student_text:
        return {"type": "error", "message": "student_input required for respond"}

    emotion = _detect_emotion(student_text)
    progress = _update_student_profile(state_key, emotion)

    current_chunk = _find_by_seq(chunks, current_seq)
    reply = _teacher_reply(student_text, current_chunk, teacher, progress)

    return {
        "type": "teacher_reply",
        "reply": reply,
        "meta": {
            "teacher_name": teacher.get("teacher_name"),
            "student_confidence": int(progress.get("confidence") or 70),
            "student_emotion": progress.get("last_emotion") or "neutral",
        },
    }