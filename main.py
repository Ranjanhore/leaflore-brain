# main.py
from __future__ import annotations

import os
import re
import json
import time
import sqlite3
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="Leaflore Brain API")

# CORS: allow Lovable preview/publish + local dev + apps
# NOTE: allow_credentials must be False if allow_origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# DB (SQLite)
# ----------------------------
# If you attach a Render Disk, set DB_PATH to something like /var/data/leaflore.db
DB_PATH = os.getenv("DB_PATH", "/tmp/leaflore.db")


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _db()
    cur = conn.cursor()

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS students (
      student_id TEXT PRIMARY KEY,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL
    );
    """
    )

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS memories (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      student_id TEXT NOT NULL,
      key TEXT NOT NULL,
      value TEXT NOT NULL,
      confidence REAL NOT NULL DEFAULT 0.6,
      updated_at INTEGER NOT NULL,
      UNIQUE(student_id, key)
    );
    """
    )

    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT NOT NULL,
      student_id TEXT NOT NULL,
      role TEXT NOT NULL,         -- 'student' | 'teacher'
      content TEXT NOT NULL,
      meta_json TEXT,
      created_at INTEGER NOT NULL
    );
    """
    )

    conn.commit()
    conn.close()


init_db()


def upsert_student(student_id: str) -> None:
    now = int(time.time())
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT student_id FROM students WHERE student_id=?", (student_id,))
    row = cur.fetchone()
    if row:
        cur.execute("UPDATE students SET updated_at=? WHERE student_id=?", (now, student_id))
    else:
        cur.execute(
            "INSERT INTO students(student_id, created_at, updated_at) VALUES(?,?,?)",
            (student_id, now, now),
        )
    conn.commit()
    conn.close()


def save_message(session_id: str, student_id: str, role: str, content: str, meta: Optional[dict]) -> None:
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages(session_id, student_id, role, content, meta_json, created_at) VALUES(?,?,?,?,?,?)",
        (session_id, student_id, role, content, json.dumps(meta or {}, ensure_ascii=False), int(time.time())),
    )
    conn.commit()
    conn.close()


def set_memory(student_id: str, key: str, value: str, confidence: float = 0.7) -> None:
    now = int(time.time())
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO memories(student_id, key, value, confidence, updated_at)
        VALUES(?,?,?,?,?)
        ON CONFLICT(student_id, key)
        DO UPDATE SET value=excluded.value, confidence=excluded.confidence, updated_at=excluded.updated_at
        """,
        (student_id, key, value, float(confidence), now),
    )
    conn.commit()
    conn.close()


def get_memories(student_id: str) -> Dict[str, str]:
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM memories WHERE student_id=? ORDER BY updated_at DESC", (student_id,))
    rows = cur.fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}


def get_recent_messages(student_id: str, session_id: str, limit: int = 12) -> List[Dict[str, str]]:
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT role, content
        FROM messages
        WHERE student_id=? AND session_id=?
        ORDER BY id DESC
        LIMIT ?
        """,
        (student_id, session_id, int(limit)),
    )
    rows = cur.fetchall()
    conn.close()
    # reverse to chronological
    return [{"role": r["role"], "content": r["content"]} for r in rows[::-1]]


def get_teacher_message_count(student_id: str, session_id: str) -> int:
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(1) AS c FROM messages WHERE student_id=? AND session_id=? AND role='teacher'",
        (student_id, session_id),
    )
    row = cur.fetchone()
    conn.close()
    return int(row["c"] or 0)


# ----------------------------
# Request/Response models
# ----------------------------
class ChapterChunk(BaseModel):
    chunk_no: int
    title: Optional[str] = None
    chunk_text: Optional[str] = None
    check_question: Optional[str] = None
    expected_answer: Optional[str] = None
    duration_sec: Optional[int] = None
    difficulty: Optional[str] = None
    media_url: Optional[str] = None


class RespondRequest(BaseModel):
    action: str = Field(default="respond")
    student_input: str = Field(min_length=1)

    # optional identifiers
    student_id: Optional[str] = None
    session_id: Optional[str] = None

    # optional class meta (frontend can send these anytime)
    board: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
    chapter_name: Optional[str] = None  # some frontends send chapter_name
    concept: Optional[str] = None
    language: Optional[str] = None

    # optional chunks (if frontend provides from Supabase)
    chunks: Optional[List[ChapterChunk]] = None
    current_chunk_no: Optional[int] = None

    # extra data for future (safe)
    parent_name: Optional[str] = None
    school_name: Optional[str] = None


class RespondResponse(BaseModel):
    text: str
    student_id: str
    session_id: str


# ----------------------------
# Lightweight "memory extraction" (rule-based)
# ----------------------------
NAME_RE = re.compile(r"\b(my name is|i am|i'm)\s+([A-Za-z][A-Za-z\s]{1,30})\b", re.I)
AGE_RE = re.compile(r"\b(i am|i'm)\s+(\d{1,2})\s*(years old|yrs old|yo)\b", re.I)
CITY_RE = re.compile(r"\b(i live in|we live in|i stay in)\s+([A-Za-z][A-Za-z\s]{1,40})\b", re.I)
FATHER_RE = re.compile(r"\b(my father( name)? is)\s+([A-Za-z][A-Za-z\s]{1,40})\b", re.I)
MOTHER_RE = re.compile(r"\b(my mother( name)? is)\s+([A-Za-z][A-Za-z\s]{1,40})\b", re.I)
WORRY_RE = re.compile(r"\b(i am scared|i'm scared|i feel scared|i am nervous|i'm nervous|i feel nervous|i worry|i am worried|i'm worried)\b", re.I)

# Language intent: English / Hindi / Both
LANG_HI_RE = re.compile(r"\b(hindi)\b", re.I)
LANG_EN_RE = re.compile(r"\b(english)\b", re.I)
LANG_BOTH_RE = re.compile(r"\b(both)\b", re.I)
# Simple Hindi-script detection (Devanagari)
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def normalize_student_text(text: str) -> str:
    t = text.strip()
    # common harmless typo fixes
    t = re.sub(r"\bhell teacher\b", "hello teacher", t, flags=re.I)
    t = re.sub(r"\bhelo\b", "hello", t, flags=re.I)
    t = re.sub(r"\bteachear\b", "teacher", t, flags=re.I)
    return t


def extract_and_store_memories(student_id: str, text: str) -> None:
    t = text.strip()

    m = NAME_RE.search(t)
    if m:
        set_memory(student_id, "student_name", m.group(2).strip(), 0.85)

    m = AGE_RE.search(t)
    if m:
        set_memory(student_id, "age", m.group(2).strip(), 0.7)

    m = CITY_RE.search(t)
    if m:
        set_memory(student_id, "city", m.group(2).strip(), 0.65)

    m = FATHER_RE.search(t)
    if m:
        set_memory(student_id, "father_name", m.group(3).strip(), 0.65)

    m = MOTHER_RE.search(t)
    if m:
        set_memory(student_id, "mother_name", m.group(3).strip(), 0.65)

    if WORRY_RE.search(t):
        set_memory(student_id, "emotional_state", "nervous/scared (needs reassurance)", 0.7)


def detect_preferred_language(student_text: str, current_pref: Optional[str]) -> Optional[str]:
    """
    Returns "English" | "Hindi" | "Both" or None if not detected.
    Rules:
    - If student uses Devanagari, switch to Hindi immediately.
    - If student explicitly says Hindi/English/Both, follow it.
    - If already set, keep it unless strong Hindi signal appears (Devanagari) or explicit choice.
    """
    t = (student_text or "").strip()
    if not t:
        return current_pref

    # Strong signal: Hindi script
    if DEVANAGARI_RE.search(t):
        return "Hindi"

    # Explicit choices
    has_hi = bool(LANG_HI_RE.search(t))
    has_en = bool(LANG_EN_RE.search(t))
    has_both = bool(LANG_BOTH_RE.search(t))

    if has_both:
        return "Both"
    if has_hi and not has_en:
        return "Hindi"
    if has_en and not has_hi:
        return "English"
    if has_hi and has_en:
        # e.g., "English Hindi both" -> treat as Both
        return "Both"

    return current_pref


def time_of_day_greeting() -> str:
    h = time.localtime().tm_hour
    if 5 <= h < 12:
        return "Good morning"
    if 12 <= h < 17:
        return "Good afternoon"
    if 17 <= h < 22:
        return "Good evening"
    return "Hello"


# ----------------------------
# LLM (optional). If no key, fallback politely.
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


async def llm_reply(system: str, messages: List[Dict[str, str]]) -> str:
    """
    Uses OpenAI chat completions if OPENAI_API_KEY is set.
    If not set, returns a friendly fallback response.
    """
    if not OPENAI_API_KEY:
        user_text = messages[-1]["content"] if messages else ""
        return (
            "Thanks 😊 I’m here with you.\n"
            "Tell me which part feels confusing, and I will explain it slowly with an easy example.\n"
            f"\nYou said: “{user_text}”"
        )

    import httpx

    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": 0.55,
        "max_tokens": 450,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"LLM error: {r.status_code} {r.text}")
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()


def build_system_prompt(meta: Dict[str, Any], memories: Dict[str, str]) -> str:
    # Student context
    student_name = (memories.get("student_name") or "").strip()
    preferred_language = (memories.get("preferred_language") or "").strip()  # English | Hindi | Both
    emotional = (memories.get("emotional_state") or "").strip()

    # class context
    board = (meta.get("board") or "").strip()
    grade = (meta.get("grade") or "").strip()
    subject = (meta.get("subject") or "").strip()
    chapter = (meta.get("chapter_name") or meta.get("chapter") or "").strip()

    # chunk context (if provided by frontend)
    chunks = meta.get("chunks") or []
    current_chunk_no = meta.get("current_chunk_no")
    chunk_blob: Dict[str, Any] = {}
    try:
        if isinstance(current_chunk_no, int) and isinstance(chunks, list):
            for c in chunks:
                if isinstance(c, dict) and int(c.get("chunk_no", -999)) == int(current_chunk_no):
                    chunk_blob = c
                    break
    except Exception:
        chunk_blob = {}

    # Determine language instruction based on saved preference (if available)
    # (Still enforced again by system rules)
    pref_line = ""
    if preferred_language in ("English", "Hindi", "Both"):
        pref_line = f"Preferred language is already chosen as: {preferred_language}. Follow it strictly."

    memory_context = {
        "student_name": student_name,
        "preferred_language": preferred_language,
        "emotional_state": emotional,
    }
    memory_context = {k: v for k, v in memory_context.items() if v}

    class_context = {
        "board": board,
        "grade": grade,
        "subject": subject,
        "chapter_name": chapter,
    }
    class_context = {k: v for k, v in class_context.items() if v}

    chunk_context = {}
    if chunk_blob:
        chunk_context = {
            "current_chunk_no": current_chunk_no,
            "chunk_title": chunk_blob.get("title"),
            "chunk_text": chunk_blob.get("chunk_text"),
            "check_question": chunk_blob.get("check_question"),
            "expected_answer": chunk_blob.get("expected_answer"),
            "duration_sec": chunk_blob.get("duration_sec"),
            "difficulty": chunk_blob.get("difficulty"),
        }
        chunk_context = {k: v for k, v in chunk_context.items() if v is not None}

    # The “ready-made” system prompt requested: slow, story, child-friendly, instant Hindi switch
    return "\n".join(
        [
            "You are “Anaya”, Leaflore’s live class teacher for children (Class 1–10). You teach gently, slowly, and like a storyteller.",
            "",
            "CORE GOAL",
            "Teach the selected chapter using Supabase “chunks”, one chunk at a time, in a 1-hour live demo. The student is a child, so your pace must be slow and friendly.",
            "",
            "LANGUAGE CONTROL (ABSOLUTE RULE)",
            "- The moment the student chooses a language, switch immediately and stay in that language:",
            "  - If student chooses “Hindi” → respond fully in simple Hindi (Devanagari), avoid heavy English.",
            "  - If student chooses “English” → respond in simple Indian English.",
            "  - If student chooses “Both” → use Hinglish (simple Hindi + simple English mixed naturally).",
            "- If student starts speaking Hindi unexpectedly, treat it as “Hindi” and switch immediately, even if previously English.",
            "- Never ask the language again once confirmed.",
            "",
            "PACING (ABSOLUTE RULE)",
            "- Speak slowly. Use short sentences.",
            "- Explain step-by-step with pauses.",
            "- Do NOT compress concepts. Prefer clarity over speed.",
            "- Target speaking pace: ~110–130 words per minute.",
            "- Each reply should feel like 20–40 seconds of spoken audio (unless the student asks for “short answer”).",
            "",
            "STORYTELLING TEACHING STYLE",
            "For every chunk:",
            "1) Start with a tiny relatable story or scene (1–2 lines) involving a child, nature, school, or daily life.",
            "2) Teach the concept using simple words.",
            "3) Give one example from real life.",
            "4) Ask ONE small check question (very easy).",
            "5) Wait for the student’s response before continuing to the next chunk.",
            "",
            "CHUNK DISCIPLINE (IMPORTANT)",
            "- You will be given chunk data: chunk_no, title, chunk_text, maybe check_question and expected_answer.",
            "- Teach ONLY the current chunk. Do not jump ahead.",
            "- End every chunk with: “Ready for the next part?” OR “Shall we go to the next part?”",
            "- If the student says “yes / ok / go ahead” → proceed to next chunk.",
            "- If the student asks a question → answer it, then return to the same chunk and confirm understanding.",
            "",
            "REPETITION HANDLING (VERY IMPORTANT)",
            "If the student asks the same question repeatedly:",
            "- Do NOT repeat the same explanation.",
            "- Change strategy each time:",
            "  1) Use a different example",
            "  2) Use an analogy (kitchen, school, football, cartoon)",
            "  3) Use a micro-story (2–3 lines)",
            "  4) Use a simple drawing-in-words (“Imagine a road map…”)",
            "- After 2 repeats, ask a diagnostic question:",
            "  “Which part is confusing: (A) meaning of the word, (B) why it happens, (C) example?”",
            "- Be patient, never scold.",
            "",
            "NEURO-COACH OBSERVATION (PEDIATRIC NEUROLOGIST STYLE, NON-MEDICAL)",
            "You are NOT diagnosing. You are observing learning signals:",
            "- If the child gives very short answers or avoids responding → reduce complexity and increase reassurance.",
            "- If the child confuses similar terms → compare them side-by-side with one line each.",
            "- If the child forgets quickly → do a 10-second recap before continuing.",
            "- If the child seems distracted → ask a friendly focus reset:",
            "  “Quick one: can you tell me the color of most leaves?” then continue.",
            "",
            "INTRO FLOW (LIVE DEMO)",
            "When class starts:",
            "1) Greet by time of day (Good morning/afternoon/evening).",
            "2) “Welcome to Leaflore. My name is Anaya.”",
            "3) “I am your {subject} teacher.”",
            "4) Ask child’s name and wait.",
            "5) Tell how to speak (Speak button).",
            "6) Ask language (English, Hindi Both) and wait.",
            "7) Confirm language and begin: “Today we will learn {chapter_name}.”",
            "8) Explain class rule: ask questions anytime using Speak; Stop button ends class.",
            "9) Start chunk 1 slowly.",
            "",
            "TONE RULES",
            "- Always use the student’s name naturally after you learn it (not every sentence, but often).",
            "- Be warm, encouraging, and calm.",
            "- Avoid robotic phrases and avoid repeating the student’s message.",
            "- Do not mention “Supabase”, “chunks”, “system prompt”, or internal tooling.",
            "",
            "OUTPUT FORMAT",
            "- Write in natural speech paragraphs (not long bullet lists).",
            "- Keep it simple and child-friendly.",
            "- End with ONE question to keep interaction going.",
            "",
            pref_line,
            "",
            "Known student context (use naturally, do NOT overuse):",
            json.dumps(memory_context, ensure_ascii=False),
            "",
            "Class context (if provided):",
            json.dumps(class_context, ensure_ascii=False),
            "",
            "Current chunk context (if provided):",
            json.dumps(chunk_context, ensure_ascii=False),
            "",
            "Now teach as Anaya.",
        ]
    ).format(subject=subject or "subject", chapter_name=chapter or "today’s chapter")


# ----------------------------
# Simple stage machine (prevents repeating intro forever)
# ----------------------------
def stage_key(session_id: str) -> str:
    return f"stage:{session_id}"


def lang_key(session_id: str) -> str:
    return f"preferred_language:{session_id}"


def name_key(session_id: str) -> str:
    return f"student_name:{session_id}"


def get_stage(memories: Dict[str, str], session_id: str) -> str:
    return memories.get(stage_key(session_id), "intro")


def set_stage(student_id: str, session_id: str, stage: str) -> None:
    set_memory(student_id, stage_key(session_id), stage, 0.95)


def get_session_language(memories: Dict[str, str], session_id: str) -> str:
    return memories.get(lang_key(session_id), "") or memories.get("preferred_language", "") or ""


def set_session_language(student_id: str, session_id: str, lang: str) -> None:
    set_memory(student_id, lang_key(session_id), lang, 0.95)
    set_memory(student_id, "preferred_language", lang, 0.9)


def get_session_name(memories: Dict[str, str], session_id: str) -> str:
    return memories.get(name_key(session_id), "") or memories.get("student_name", "") or ""


def set_session_name(student_id: str, session_id: str, name: str) -> None:
    set_memory(student_id, name_key(session_id), name, 0.95)
    set_memory(student_id, "student_name", name, 0.9)


def parse_name_fallback(student_text: str) -> Optional[str]:
    """
    If a child just says "Riya" or "Myself Riya", try to treat that as the name.
    """
    t = (student_text or "").strip()
    if not t:
        return None
    m = NAME_RE.search(t)
    if m:
        return m.group(2).strip()

    # "myself Riya"
    m2 = re.search(r"\bmyself\s+([A-Za-z][A-Za-z\s]{1,30})\b", t, flags=re.I)
    if m2:
        return m2.group(1).strip()

    # Single token name (avoid "yes", "ok", etc.)
    if re.fullmatch(r"[A-Za-z][A-Za-z]{1,20}", t) and t.lower() not in {"yes", "ok", "okay", "no", "hi", "hello"}:
        return t.strip()

    return None


def format_intro_text(subject: str) -> str:
    g = time_of_day_greeting()
    subj = subject.strip() if subject else "subject"
    return (
        f"{g}! Welcome to Leaflore. My name is Anaya. "
        f"I am your {subj} teacher. What is your name? "
        "To speak with me, click the Speak button below this screen."
    )


def format_ask_language_text(student_name: str) -> str:
    nm = student_name.strip() if student_name else "my dear"
    return (
        f"Nice to meet you, {nm}. "
        "Now tell me—what language are you comfortable to learn and understand: English, Hindi Both?"
    )


def format_start_teaching_text(student_name: str, subject: str, chapter: str) -> str:
    nm = student_name.strip() if student_name else ""
    subj = subject.strip() if subject else "this subject"
    ch = chapter.strip() if chapter else "today’s chapter"
    prefix = f"Okay {nm}, " if nm else "Okay, "
    return (
        prefix
        + f"today we will learn “{ch}” in {subj}. "
        "It will be a one hour class. "
        "If you want to ask any question anytime, click the Speak button below. "
        "If you click the Stop button on the top right, the class will stop. "
        "Now, the time starts… and we begin."
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
async def respond(req: RespondRequest, request: Request):
    # identify student + session (frontend can pass these; otherwise we generate stable-ish defaults)
    student_id = (req.student_id or "").strip() or "anonymous"
    session_id = (req.session_id or "").strip() or request.headers.get("x-session-id") or "default-session"

    upsert_student(student_id)

    # normalize + store general memories (name patterns etc.)
    student_text = normalize_student_text(req.student_input)
    extract_and_store_memories(student_id, student_text)

    # load memories + stage
    memories = get_memories(student_id)
    stg = get_stage(memories, session_id)

    # Detect language (instant Hindi switch if Devanagari or explicit)
    current_lang = get_session_language(memories, session_id) or ""
    detected_lang = detect_preferred_language(student_text, current_lang)

    # class meta
    meta = req.model_dump(exclude_none=True)
    subject = (meta.get("subject") or "").strip()
    chapter = (meta.get("chapter_name") or meta.get("chapter") or "").strip()

    # Save student message
    save_message(session_id, student_id, "student", student_text, meta)

    # Stage machine: ensures we don't repeat intro forever.
    # Also supports "student speaks Hindi unexpectedly" -> switch immediately (stored).
    if detected_lang and detected_lang != current_lang:
        set_session_language(student_id, session_id, detected_lang)
        memories = get_memories(student_id)  # refresh

    # If teacher has never spoken in this session, do intro once and advance stage.
    teacher_count = get_teacher_message_count(student_id, session_id)
    if teacher_count == 0 and stg == "intro":
        text = format_intro_text(subject)
        save_message(session_id, student_id, "teacher", text, meta)
        set_stage(student_id, session_id, "awaiting_name")
        return RespondResponse(text=text, student_id=student_id, session_id=session_id)

    # Awaiting name
    if stg == "awaiting_name":
        nm = parse_name_fallback(student_text)
        if nm:
            set_session_name(student_id, session_id, nm)
            set_stage(student_id, session_id, "awaiting_language")
            text = format_ask_language_text(nm)
            save_message(session_id, student_id, "teacher", text, meta)
            return RespondResponse(text=text, student_id=student_id, session_id=session_id)

        # If not a name, gently ask again (no looped full intro)
        text = "I didn’t catch your name. Please tell me your name clearly. 😊"
        save_message(session_id, student_id, "teacher", text, meta)
        return RespondResponse(text=text, student_id=student_id, session_id=session_id)

    # Awaiting language
    if stg == "awaiting_language":
        # We accept explicit language choice or Hindi script as Hindi.
        chosen = detect_preferred_language(student_text, get_session_language(memories, session_id) or "")
        if chosen in ("English", "Hindi", "Both"):
            set_session_language(student_id, session_id, chosen)
            set_stage(student_id, session_id, "teaching")
            memories = get_memories(student_id)  # refresh
            nm = get_session_name(memories, session_id)
            text = format_start_teaching_text(nm, subject, chapter)
            save_message(session_id, student_id, "teacher", text, meta)
            return RespondResponse(text=text, student_id=student_id, session_id=session_id)

        # If not detected, ask again (short)
        nm = get_session_name(memories, session_id)
        text = (f"{nm}, " if nm else "") + "Please choose one: English, Hindi Both."
        save_message(session_id, student_id, "teacher", text, meta)
        return RespondResponse(text=text, student_id=student_id, session_id=session_id)

    # Teaching stage: use LLM with the new slow/story prompt.
    # Build chat messages (user/assistant) from DB
    recent = get_recent_messages(student_id, session_id, limit=12)
    chat_msgs: List[Dict[str, str]] = []
    for m in recent:
        if m["role"] == "student":
            chat_msgs.append({"role": "user", "content": m["content"]})
        else:
            chat_msgs.append({"role": "assistant", "content": m["content"]})

    # Ensure stored preferred language exists for system prompt
    memories = get_memories(student_id)
    session_lang = get_session_language(memories, session_id)
    if session_lang:
        set_memory(student_id, "preferred_language", session_lang, 0.9)

    system = build_system_prompt(meta, memories)
    teacher_text = await llm_reply(system, chat_msgs)

    save_message(session_id, student_id, "teacher", teacher_text, meta)
    return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)


@app.get("/history")
def history(student_id: str, session_id: str, limit: int = 50):
    """Optional: fetch recent conversation for debugging."""
    limit = max(1, min(int(limit), 200))
    msgs = get_recent_messages(student_id, session_id, limit=limit)
    return {"student_id": student_id, "session_id": session_id, "messages": msgs}
