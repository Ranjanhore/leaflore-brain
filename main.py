from __future__ import annotations

import os
import re
import time
import random
import sqlite3
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# -----------------------------
# Config
# -----------------------------
APP_NAME = "Leaflore Brain API"

# Use Render persistent disk if mounted; otherwise local file.
DATA_DIR = os.environ.get("DATA_DIR") or os.environ.get("RENDER_DISK_PATH") or "."
DB_PATH = os.path.join(DATA_DIR, "leaflore.db")

# Allow Lovable preview/publish domains + localhost.
# This regex covers:
# - *.lovable.app
# - *.lovableproject.com
# - localhost / 127.0.0.1 (any port)
ALLOW_ORIGIN_REGEX = os.environ.get(
    "ALLOW_ORIGIN_REGEX",
    r"^https?://([a-z0-9-]+\.)*(lovable\.app|lovableproject\.com)(:\d+)?$|^http://localhost(:\d+)?$|^http://127\.0\.0\.1(:\d+)?$",
)

# Keep a small rolling context per student for “more human” replies
CONTEXT_TURNS = int(os.environ.get("CONTEXT_TURNS", "10"))


# -----------------------------
# DB Helpers (SQLite)
# -----------------------------
def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    con = db()
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT,
            grade TEXT,
            board TEXT,
            preferred_language TEXT,
            created_at INTEGER,
            updated_at INTEGER
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            role TEXT,               -- 'student' | 'teacher'
            content TEXT,
            subject TEXT,
            chapter TEXT,
            created_at INTEGER
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            key TEXT,
            value TEXT,
            confidence REAL,
            created_at INTEGER
        )
        """
    )

    con.commit()
    con.close()


def upsert_student(student_id: str, name: Optional[str], grade: Optional[str], board: Optional[str], language: Optional[str]) -> None:
    now = int(time.time())
    con = db()
    cur = con.cursor()
    cur.execute("SELECT student_id FROM students WHERE student_id = ?", (student_id,))
    exists = cur.fetchone() is not None

    if not exists:
        cur.execute(
            """
            INSERT INTO students (student_id, name, grade, board, preferred_language, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (student_id, name, grade, board, language, now, now),
        )
    else:
        # Only overwrite non-empty fields
        cur.execute("SELECT name, grade, board, preferred_language FROM students WHERE student_id = ?", (student_id,))
        row = cur.fetchone()
        cur.execute(
            """
            UPDATE students
            SET name = COALESCE(NULLIF(?, ''), ?),
                grade = COALESCE(NULLIF(?, ''), ?),
                board = COALESCE(NULLIF(?, ''), ?),
                preferred_language = COALESCE(NULLIF(?, ''), ?),
                updated_at = ?
            WHERE student_id = ?
            """,
            (
                name or "", row["name"],
                grade or "", row["grade"],
                board or "", row["board"],
                language or "", row["preferred_language"],
                now,
                student_id,
            ),
        )

    con.commit()
    con.close()


def add_message(student_id: str, role: str, content: str, subject: Optional[str], chapter: Optional[str]) -> None:
    con = db()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO messages (student_id, role, content, subject, chapter, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (student_id, role, content, subject, chapter, int(time.time())),
    )
    con.commit()
    con.close()


def get_recent_messages(student_id: str, limit: int = CONTEXT_TURNS) -> List[Dict[str, Any]]:
    con = db()
    cur = con.cursor()
    cur.execute(
        """
        SELECT role, content, subject, chapter, created_at
        FROM messages
        WHERE student_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (student_id, limit),
    )
    rows = cur.fetchall()
    con.close()
    return [dict(r) for r in reversed(rows)]


def upsert_fact(student_id: str, key: str, value: str, confidence: float = 0.75) -> None:
    # Keep it simple: allow multiple facts over time; newest is “latest”
    con = db()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO facts (student_id, key, value, confidence, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (student_id, key, value, confidence, int(time.time())),
    )
    con.commit()
    con.close()


def get_latest_facts(student_id: str, keys: Optional[List[str]] = None, limit: int = 20) -> Dict[str, str]:
    con = db()
    cur = con.cursor()
    if keys:
        placeholders = ",".join(["?"] * len(keys))
        cur.execute(
            f"""
            SELECT key, value
            FROM facts
            WHERE student_id = ? AND key IN ({placeholders})
            ORDER BY id DESC
            """,
            (student_id, *keys),
        )
    else:
        cur.execute(
            """
            SELECT key, value
            FROM facts
            WHERE student_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (student_id, limit),
        )
    rows = cur.fetchall()
    con.close()

    latest: Dict[str, str] = {}
    for r in rows:
        k = r["key"]
        if k not in latest:
            latest[k] = r["value"]
    return latest


# -----------------------------
# Lightweight “Memory” Extraction
# (safe + simple heuristics)
# -----------------------------
NAME_PAT = re.compile(r"\b(my name is|i am|i'm)\s+([A-Za-z][A-Za-z\s]{1,30})\b", re.IGNORECASE)
PARENT_PAT = re.compile(r"\b(my (mom|mother|dad|father)('?s)? name is)\s+([A-Za-z][A-Za-z\s]{1,30})\b", re.IGNORECASE)
FEEL_PAT = re.compile(r"\b(i feel|i am feeling|i'm feeling)\s+([a-z\s]{2,30})\b", re.IGNORECASE)


def extract_facts_from_student_text(text: str) -> List[Dict[str, Any]]:
    found = []

    m = NAME_PAT.search(text)
    if m:
        name = m.group(2).strip()
        if 2 <= len(name) <= 40:
            found.append({"key": "student_name", "value": name, "confidence": 0.8})

    p = PARENT_PAT.search(text)
    if p:
        parent_name = p.group(4).strip()
        rel = p.group(2).lower()
        found.append({"key": f"{rel}_name", "value": parent_name, "confidence": 0.75})

    f = FEEL_PAT.search(text)
    if f:
        feeling = f.group(2).strip()
        found.append({"key": "student_feeling", "value": feeling, "confidence": 0.7})

    # Soft “fear / anxiety” signal
    if re.search(r"\b(scared|afraid|anxious|worried|panic|nervous)\b", text, re.IGNORECASE):
        found.append({"key": "emotion_flag", "value": "anxiety", "confidence": 0.7})

    # “Don’t understand” signal
    if re.search(r"\b(i can'?t understand|i don'?t get it|confused|too hard)\b", text, re.IGNORECASE):
        found.append({"key": "learning_flag", "value": "needs_simpler_explanation", "confidence": 0.75})

    return found


# -----------------------------
# Response “Brain” (human-ish)
# -----------------------------
HUMBLE_OPENERS = [
    "Got you.",
    "I’m with you.",
    "Thanks for telling me.",
    "That’s completely okay.",
    "No worries—let’s do it together.",
]
SOOTHERS = [
    "You’re safe to ask anything here.",
    "I won’t be upset—learning takes time.",
    "It’s normal to feel stuck sometimes.",
    "We’ll go step by step.",
]
FOLLOWUP_STYLES = [
    "Tell me what part felt confusing—was it the meaning, the steps, or an example?",
    "What do you already know about it, even a tiny bit?",
    "Do you want a simple example first, or a quick explanation first?",
    "If you had to guess, what would you say it means?",
]

def pick(arr: List[str]) -> str:
    return random.choice(arr)

def infer_topic(student_text: str, subject: Optional[str]) -> str:
    # If subject provided, use it; else infer broadly
    if subject and subject.strip():
        return subject.strip()
    t = student_text.lower()
    if any(w in t for w in ["math", "algebra", "fraction", "divide", "multiply", "equation"]):
        return "Math"
    if any(w in t for w in ["photosynthesis", "plant", "biology", "cell", "science"]):
        return "Science"
    if any(w in t for w in ["history", "king", "battle", "mughal", "independence"]):
        return "Social Science"
    if any(w in t for w in ["grammar", "meaning", "sentence", "english", "hindi"]):
        return "Language"
    return "your topic"

def natural_reply(student_text: str, student_id: str, subject: Optional[str], chapter: Optional[str]) -> str:
    facts = get_latest_facts(student_id, keys=["student_name", "student_feeling", "emotion_flag", "learning_flag"])
    name = facts.get("student_name")
    anxiety = facts.get("emotion_flag") == "anxiety"
    needs_simple = facts.get("learning_flag") == "needs_simpler_explanation"

    topic = infer_topic(student_text, subject)

    # Small talk / personal questions: answer naturally, then gently steer back
    if re.search(r"\b(had lunch|had dinner|what did you eat|lunch)\b", student_text, re.IGNORECASE):
        base = "I had something simple, thanks for asking 🙂"
        steer = f"Want to learn {topic} now, or do you want to tell me what you’re working on today?"
        return f"{base} {steer}"

    if re.search(r"\b(what('?s)? your name|who are you)\b", student_text, re.IGNORECASE):
        who = "I’m your Leaflore teacher—here with you like a calm guide."
        greet = f" What should I call you?" if not name else f" Hi {name} 🙂 What are we learning today?"
        return who + greet

    # Student fear / reassurance
    if re.search(r"\b(angry|scold|shout|be mad)\b", student_text, re.IGNORECASE) or re.search(r"\b(can't understand|dont understand|too hard|confused)\b", student_text, re.IGNORECASE):
        line1 = pick(HUMBLE_OPENERS)
        line2 = pick(SOOTHERS)
        line3 = "Let’s make it easy—one tiny step at a time."
        prompt = pick(FOLLOWUP_STYLES)
        if name:
            return f"{line1} {line2} {line3} {name}, {prompt}"
        return f"{line1} {line2} {line3} {prompt}"

    # If anxious signal exists, keep tone extra gentle
    if anxiety:
        return f"{pick(HUMBLE_OPENERS)} {pick(SOOTHERS)} Let’s start very small: {pick(FOLLOWUP_STYLES)}"

    # Default “good teacher chat” response: acknowledge + ask a clarifying question + offer options
    opener = pick(HUMBLE_OPENERS)
    if needs_simple:
        return (
            f"{opener} I’ll keep it super simple.\n"
            f"Tell me one thing: what exactly do you need—meaning, steps, or an example?"
        )

    # If student asks a direct “topic” question: respond briefly + ask follow-up
    if student_text.strip().endswith("?"):
        return (
            f"{opener} Let’s answer that clearly.\n"
            f"Before I explain, tell me: is this from {topic} homework/class, or just curiosity?"
        )

    return (
        f"{opener} Tell me what you want to learn in {topic} today.\n"
        f"If you paste the question, I’ll explain it like a friend—and we’ll practice once together."
    )


# -----------------------------
# API Models
# -----------------------------
class RespondRequest(BaseModel):
    action: str = Field(default="respond")
    student_input: str

    # Optional “context”
    student_id: Optional[str] = Field(default="demo")
    board: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
    language: Optional[str] = None

    # Optional signals (frontend can send)
    signals: Optional[Dict[str, Any]] = None


class RespondResponse(BaseModel):
    text: str
    student_id: str
    subject: Optional[str] = None
    chapter: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# init db at import time (Render-friendly)
init_db()


@app.get("/")
def root():
    return {"service": APP_NAME, "health": "/health", "respond": "/respond"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/respond", response_model=RespondResponse)
def respond(req: RespondRequest):
    if req.action != "respond":
        # Keep it strict to avoid mismatches
        raise HTTPException(status_code=400, detail={"error": "Invalid action", "expected": "respond"})

    student_id = (req.student_id or "demo").strip() or "demo"
    student_text = (req.student_input or "").strip()
    if not student_text:
        raise HTTPException(status_code=400, detail={"error": "student_input is required"})

    # Save/update student
    upsert_student(
        student_id=student_id,
        name=None,
        grade=req.grade,
        board=req.board,
        language=req.language,
    )

    # Extract + store facts
    for fact in extract_facts_from_student_text(student_text):
        upsert_fact(student_id, fact["key"], fact["value"], float(fact["confidence"]))

    # Save student message
    add_message(student_id, "student", student_text, req.subject, req.chapter)

    # Generate reply (human-ish)
    reply_text = natural_reply(student_text, student_id, req.subject, req.chapter)

    # Save teacher message
    add_message(student_id, "teacher", reply_text, req.subject, req.chapter)

    # Meta for debugging
    recent = get_recent_messages(student_id, limit=CONTEXT_TURNS)
    facts = get_latest_facts(student_id, limit=10)

    return RespondResponse(
        text=reply_text,
        student_id=student_id,
        subject=req.subject,
        chapter=req.chapter,
        meta={
            "context_turns": len(recent),
            "latest_facts": facts,
        },
    )
