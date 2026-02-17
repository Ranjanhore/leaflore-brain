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

def set_memory(student_id: str, key: str, value: str, confidence: float = 0.75) -> None:
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

def get_recent_messages(student_id: str, session_id: str, limit: int = 14) -> List[Dict[str, str]]:
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

# ----------------------------
# Models
# ----------------------------
class RespondRequest(BaseModel):
    # actions:
    # - "start_class": teacher speaks FIRST (welcome flow)
    # - "respond": normal back-and-forth teaching
    action: str = Field(default="respond")
    student_input: str = Field(default="")

    student_id: Optional[str] = None
    session_id: Optional[str] = None

    board: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
    concept: Optional[str] = None
    language: Optional[str] = None  # optionally passed by frontend
    parent_name: Optional[str] = None
    school_name: Optional[str] = None

class RespondResponse(BaseModel):
    text: str
    student_id: str
    session_id: str

# ----------------------------
# Memory extraction (rule-based)
# ----------------------------
NAME_RE = re.compile(r"\b(my name is|i am|i'm)\s+([A-Za-z][A-Za-z\s]{1,30})\b", re.I)
LANG_RE = re.compile(r"\b(english|hindi|both|hinglish)\b", re.I)

def normalize_student_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\bhell teacher\b", "hello teacher", t, flags=re.I)
    t = re.sub(r"\bhelo\b", "hello", t, flags=re.I)
    t = re.sub(r"\bteachear\b", "teacher", t, flags=re.I)
    return t

def extract_and_store_memories(student_id: str, text: str) -> None:
    t = (text or "").strip()

    m = NAME_RE.search(t)
    if m:
        set_memory(student_id, "student_name", m.group(2).strip(), 0.85)

    # language preference extraction (only if student answered it)
    ml = LANG_RE.search(t)
    if ml:
        raw = ml.group(1).lower()
        if raw == "hinglish":
            raw = "both"
        # store normalized
        if raw in ("english", "hindi", "both"):
            set_memory(student_id, "preferred_language", raw, 0.85)

def time_greeting() -> str:
    hr = time.localtime().tm_hour
    if hr < 12:
        return "Good morning"
    if hr < 17:
        return "Good afternoon"
    return "Good evening"

def norm_lang_pref(v: Optional[str]) -> Optional[str]:
    if not v:
        return None
    x = v.strip().lower()
    if "hinglish" in x or "both" in x:
        return "both"
    if "hindi" in x:
        return "hindi"
    if "english" in x:
        return "english"
    return None

# ----------------------------
# LLM (optional)
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

async def llm_reply(system: str, messages: List[Dict[str, str]]) -> str:
    if not OPENAI_API_KEY:
        # fallback
        user_text = messages[-1]["content"] if messages else ""
        return (
            "Hello 😊 I’m Anaya, your Leaflore teacher.\n"
            "Tell me your name and which language you prefer (English / Hindi / Both), and we’ll start."
            f"\n\nYou said: “{user_text}”"
        )

    import httpx

    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": 0.7,
        "max_tokens": 450,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=35.0) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers
