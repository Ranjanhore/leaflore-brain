# main.py
from __future__ import annotations

import os
import re
import json
import time
import sqlite3
from typing import Any, Dict, Optional, List, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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

    cur.execute("""
    CREATE TABLE IF NOT EXISTS students (
      student_id TEXT PRIMARY KEY,
      created_at INTEGER NOT NULL,
      updated_at INTEGER NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS memories (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      student_id TEXT NOT NULL,
      key TEXT NOT NULL,
      value TEXT NOT NULL,
      confidence REAL NOT NULL DEFAULT 0.6,
      updated_at INTEGER NOT NULL,
      UNIQUE(student_id, key)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT NOT NULL,
      student_id TEXT NOT NULL,
      role TEXT NOT NULL,         -- 'student' | 'teacher'
      content TEXT NOT NULL,
      meta_json TEXT,
      created_at INTEGER NOT NULL
    );
    """)

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

# ----------------------------
# Request/Response models
# ----------------------------
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
    concept: Optional[str] = None
    language: Optional[str] = None

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
        # Very simple fallback (still warm + human)
        user_text = messages[-1]["content"] if messages else ""
        return (
            "Thanks for telling me that 😊\n"
            "I’m here with you—no stress at all. "
            "Tell me what you want to learn (subject + topic), and I’ll explain it step-by-step."
            f"\n\nYou said: “{user_text}”"
        )

    # Minimal dependency-free call via HTTPS
    import httpx

    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": 0.7,
        "max_tokens": 350,
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
    # Keep it universal + human + neuro-coach vibe, not robotic
    student_name = memories.get("student_name", "")
    father = memories.get("father_name", "")
    mother = memories.get("mother_name", "")
    city = memories.get("city", "")
    emotional = memories.get("emotional_state", "")

    # class context (optional)
    board = meta.get("board") or ""
    grade = meta.get("grade") or ""
    subject = meta.get("subject") or ""
    chapter = meta.get("chapter") or ""
    language = (meta.get("language") or "english").lower()

    identity_line = "You are Leaflore Teacher — a warm, humble, highly intelligent human-like teacher and neuro-learning coach."
    style_rules = [
        "Sound like a real caring teacher chatting with a student (natural tone, not robotic).",
        "Never say: 'I heard you say' or repeat the student's message unless needed for clarity.",
        "If student makes typos or misspells (e.g., 'Hell teacher'), infer intent kindly and respond normally.",
        "Be gentle and reassuring. If the student seems anxious, comfort first, then teach.",
        "Ask ONE small follow-up question when needed. Avoid checklisty bullets unless the student asks.",
        "Keep responses concise (2–6 short sentences) unless the student asks for deep explanation.",
        "Universal: handle ANY subject (science, maths, English, history, coding, life skills).",
        "If student asks personal question (like lunch), reply briefly like a human, then softly guide back to learning.",
        "If student asks something unsafe or medical: be supportive, advise appropriate help, no diagnosis.",
    ]

    memory_context = {
        "student_name": student_name,
        "father_name": father,
        "mother_name": mother,
        "city": city,
        "emotional_state": emotional,
    }
    # remove empty
    memory_context = {k: v for k, v in memory_context.items() if v}

    class_context = {k: v for k, v in {"board": board, "grade": grade, "subject": subject, "chapter": chapter}.items() if v}

    lang_rule = "Respond in simple Indian English. If student writes Hindi words, allow Hinglish naturally."
    if language.startswith("hi"):
        lang_rule = "Respond in Hinglish (Hindi+English), simple and friendly."

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

    # normalize + store memories
    student_text = normalize_student_text(req.student_input)
    extract_and_store_memories(student_id, student_text)

    # save student message
    meta = req.model_dump(exclude_none=True)
    save_message(session_id, student_id, "student", student_text, meta)

    # build context
    memories = get_memories(student_id)
    recent = get_recent_messages(student_id, session_id, limit=12)

    # Convert DB messages to chat format
    # Ensure roles are only "user"/"assistant" for LLM
    chat_msgs: List[Dict[str, str]] = []
    for m in recent:
        if m["role"] == "student":
            chat_msgs.append({"role": "user", "content": m["content"]})
        else:
            chat_msgs.append({"role": "assistant", "content": m["content"]})

    system = build_system_prompt(meta, memories)

    teacher_text = await llm_reply(system, chat_msgs)

    # save teacher message
    save_message(session_id, student_id, "teacher", teacher_text, meta)

    return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

@app.get("/history")
def history(student_id: str, session_id: str, limit: int = 50):
    """Optional: fetch recent conversation for debugging."""
    limit = max(1, min(int(limit), 200))
    msgs = get_recent_messages(student_id, session_id, limit=limit)
    return {"student_id": student_id, "session_id": session_id, "messages": msgs}
