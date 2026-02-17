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
    return [{"role": r["role"], "content": r["content"]} for r in rows[::-1]]


# --- New helper getters/setters for onboarding state ---
def get_memory(student_id: str, key: str) -> Optional[str]:
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT value FROM memories WHERE student_id=? AND key=? LIMIT 1", (student_id, key))
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else None


def set_stage(student_id: str, stage: str) -> None:
    set_memory(student_id, "onboarding_stage", stage, 0.95)


def get_stage(student_id: str) -> str:
    return (get_memory(student_id, "onboarding_stage") or "").strip() or "none"


def time_greeting() -> str:
    # Simple local greeting (server time). If you want India time always, set TZ on server or use pytz.
    hour = time.localtime().tm_hour
    if 5 <= hour < 12:
        return "Good morning"
    if 12 <= hour < 17:
        return "Good afternoon"
    if 17 <= hour < 22:
        return "Good evening"
    return "Hello"


# ----------------------------
# Request/Response models
# ----------------------------
class RespondRequest(BaseModel):
    action: str = Field(default="respond")  # "start_class" | "respond"
    student_input: str = Field(default="")  # can be empty for start_class

    # optional identifiers
    student_id: Optional[str] = None
    session_id: Optional[str] = None

    # optional class meta (frontend can send these anytime)
    board: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
    concept: Optional[str] = None
    language: Optional[str] = None  # optional "english"/"hindi"/"both"/"bangla" etc.

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
WORRY_RE = re.compile(
    r"\b(i am scared|i'm scared|i feel scared|i am nervous|i'm nervous|i feel nervous|i worry|i am worried|i'm worried)\b",
    re.I,
)

# Language detection for onboarding reply (supports English/Hindi/Both/Bangla)
LANG_EN_RE = re.compile(r"\b(english|eng)\b", re.I)
LANG_HI_RE = re.compile(r"\b(hindi|hin)\b", re.I)
LANG_BN_RE = re.compile(r"\b(bangla|bengali|বাংলা)\b", re.I)
LANG_BOTH_RE = re.compile(r"\b(both|hinglish|mix)\b", re.I)


def normalize_student_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\bhell teacher\b", "hello teacher", t, flags=re.I)
    t = re.sub(r"\bhelo\b", "hello", t, flags=re.I)
    t = re.sub(r"\bteachear\b", "teacher", t, flags=re.I)
    return t


def extract_preferred_language(student_id: str, text: str) -> None:
    t = (text or "").strip()

    # Order matters: "both" should win over english/hindi if present
    if LANG_BOTH_RE.search(t):
        set_memory(student_id, "preferred_language", "both", 0.9)
        return
    if LANG_BN_RE.search(t):
        set_memory(student_id, "preferred_language", "bangla", 0.9)
        return
    if LANG_HI_RE.search(t):
        set_memory(student_id, "preferred_language", "hindi", 0.9)
        return
    if LANG_EN_RE.search(t):
        set_memory(student_id, "preferred_language", "english", 0.9)
        return


def extract_and_store_memories(student_id: str, text: str) -> None:
    t = (text or "").strip()

    m = NAME_RE.search(t)
    if m:
        set_memory(student_id, "student_name", m.group(2).strip(), 0.8)

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

    # language choice extraction (English/Hindi/Both/Bangla)
    extract_preferred_language(student_id, t)


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
            "Thanks for telling me that 😊\n"
            "I’m here with you—no stress at all. "
            "Tell me what you want to learn (subject + topic), and I’ll explain it step-by-step."
            f"\n\nYou said: “{user_text}”"
        )

    import httpx

    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": 0.7,
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
    student_name = memories.get("student_name", "")
    father = memories.get("father_name", "")
    mother = memories.get("mother_name", "")
    city = memories.get("city", "")
    emotional = memories.get("emotional_state", "")
    preferred_lang = (memories.get("preferred_language") or "").lower().strip()

    board = meta.get("board") or ""
    grade = meta.get("grade") or ""
    subject = meta.get("subject") or ""
    chapter = meta.get("chapter") or ""

    identity_line = "You are Leaflore Teacher — a warm, humble, highly intelligent human-like teacher named Anaya."

    style_rules = [
        "Sound like a real caring teacher chatting with a student (natural tone, not robotic).",
        "Use the student's name naturally when you know it.",
        "Never say: 'I heard you say' or repeat the student's message unless needed for clarity.",
        "Be gentle and reassuring. If the student seems anxious, comfort first, then teach.",
        "Ask ONE small follow-up question when needed. Avoid checklisty bullets unless the student asks.",
        "Keep responses concise (2–8 short sentences) unless the student asks for deep explanation.",
        "Universal: handle ANY subject (science, maths, English, history, coding, life skills).",
        "If student asks personal question (like lunch), reply briefly like a human, then softly guide back to learning.",
        "If student asks something unsafe or medical: be supportive, advise appropriate help, no diagnosis.",
        "When teaching a chapter: go step-by-step, concept-by-concept, like chunks. After each chunk, ask one quick check question.",
    ]

    # Language rules including Bangla
    if preferred_lang == "hindi":
        lang_rule = "Respond in Hinglish (Hindi+English), simple and friendly."
    elif preferred_lang == "bangla":
        lang_rule = "Respond in Bangla (বাংলা) with very simple words; you may mix a little English for science terms."
    elif preferred_lang == "both":
        lang_rule = "Respond in a mix (English + Hindi/Bangla as needed), simple and friendly."
    else:
        lang_rule = "Respond in simple Indian English. If student writes Hindi/Bangla words, allow naturally."

    memory_context = {
        "student_name": student_name,
        "father_name": father,
        "mother_name": mother,
        "city": city,
        "emotional_state": emotional,
        "preferred_language": preferred_lang,
    }
    memory_context = {k: v for k, v in memory_context.items() if v}

    class_context = {k: v for k, v in {"board": board, "grade": grade, "subject": subject, "chapter": chapter}.items() if v}

    return "\n".join(
        [
            identity_line,
            lang_rule,
            "Style rules:",
            *[f"- {r}" for r in style_rules],
            "",
            "Known student context (use naturally, do NOT overuse):",
            json.dumps(memory_context, ensure_ascii=False),
            "",
            "Class context (if provided):",
            json.dumps(class_context, ensure_ascii=False),
            "",
            "Goal: Help the student learn with confidence. Be kind, specific, and human.",
        ]
    )


# ----------------------------
# Onboarding (deterministic state machine)
# ----------------------------
def onboarding_start_text(subject: str) -> str:
    subject = subject.strip() or "your subject"
    return (
        f"{time_greeting()}! Welcome to Leaflore.\n"
        f"My name is Anaya. I am your {subject} teacher.\n\n"
        "What is your name?\n"
        "To speak with me, click the Speak button below this screen."
    )


def onboarding_ask_language_text(student_name: str) -> str:
    name = (student_name or "dear student").strip()
    return (
        f"Lovely, {name}.\n"
        "Which language are you comfortable with — English, Hindi, Bangla, or Both?"
    )


def onboarding_class_start_text(student_name: str, pref_lang: str, chapter: str) -> str:
    name = (student_name or "dear student").strip()
    chapter = chapter.strip() or "today’s chapter"

    pref_lang = (pref_lang or "").lower().strip()
    if pref_lang == "english":
        lang_line = "Great — we’ll learn in English."
    elif pref_lang == "hindi":
        lang_line = "बहुत बढ़िया — हम हिंदी में सीखेंगे।"
    elif pref_lang == "bangla":
        lang_line = "খুব ভালো — আমরা বাংলায় শিখবো।"
    elif pref_lang == "both":
        lang_line = "Awesome — we’ll learn in a mix (English + Hindi/Bangla)."
    else:
        lang_line = "Great — we’ll learn in a simple, comfortable way."

    return (
        f"{time_greeting()}, {name}! {lang_line}\n\n"
        f"Today we will learn: {chapter}.\n"
        "It will be a one hour class.\n\n"
        "Before we start: whenever you want to ask a question, click the Speak button below this screen.\n"
        "If you want to stop the class, click the Stop button on the top-right. "
        "If you stop in between the class, the class ends and won’t restart from the beginning.\n\n"
        "So let’s start learning — time starts now.\n\n"
        f"✅ First step: Tell me what you already know about **{chapter}** in one line."
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
    # identify student + session (frontend should pass these; otherwise defaults)
    student_id = (req.student_id or "").strip() or "anonymous"
    session_id = (req.session_id or "").strip() or request.headers.get("x-session-id") or "default-session"

    upsert_student(student_id)

    meta = req.model_dump(exclude_none=True)
    action = (req.action or "respond").strip().lower()

    # always keep class meta fresh (optional)
    subject = (meta.get("subject") or "").strip() or "your subject"
    chapter = (meta.get("chapter") or "").strip() or "today’s chapter"

    # --- Start class (deterministic) ---
    if action == "start_class":
        # Save a marker student message (for history)
        save_message(session_id, student_id, "student", "[Start Class clicked]", meta)

        # Move onboarding to awaiting_name every time Start Class is pressed
        set_stage(student_id, "awaiting_name")

        teacher_text = onboarding_start_text(subject)
        save_message(session_id, student_id, "teacher", teacher_text, meta)
        return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

    # --- Normal respond ---
    student_text = normalize_student_text(req.student_input or "")
    if not student_text:
        raise HTTPException(status_code=400, detail="student_input is required for action=respond")

    # store memories (name/lang/etc)
    extract_and_store_memories(student_id, student_text)

    # save student message
    save_message(session_id, student_id, "student", student_text, meta)

    # deterministic onboarding transitions (never go silent)
    stage = get_stage(student_id)
    memories = get_memories(student_id)

    student_name = (memories.get("student_name") or "").strip()
    pref_lang = (memories.get("preferred_language") or "").strip().lower()

    # If we were awaiting name and now have it -> ask language
    if stage in ("awaiting_name", "none") and student_name:
        set_stage(student_id, "awaiting_language")
        stage = "awaiting_language"

    # If still awaiting name, ask again
    if stage == "awaiting_name" and not student_name:
        teacher_text = onboarding_start_text(subject)
        save_message(session_id, student_id, "teacher", teacher_text, meta)
        return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

    # If awaiting language and not set, ask again (English/Hindi/Bangla/Both)
    if stage == "awaiting_language" and pref_lang not in ("english", "hindi", "bangla", "both"):
        teacher_text = onboarding_ask_language_text(student_name)
        save_message(session_id, student_id, "teacher", teacher_text, meta)
        return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

    # If language is set while awaiting_language -> immediately start class
    if stage == "awaiting_language" and pref_lang in ("english", "hindi", "bangla", "both"):
        set_stage(student_id, "teaching")
        teacher_text = onboarding_class_start_text(student_name, pref_lang, chapter)
        save_message(session_id, student_id, "teacher", teacher_text, meta)
        return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

    # --- Teaching / general chat via LLM (after onboarding) ---
    # Build chat history for LLM
    recent = get_recent_messages(student_id, session_id, limit=12)
    chat_msgs: List[Dict[str, str]] = []
    for m in recent:
        if m["role"] == "student":
            chat_msgs.append({"role": "user", "content": m["content"]})
        else:
            chat_msgs.append({"role": "assistant", "content": m["content"]})

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
