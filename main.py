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
# DB
# ----------------------------
DB_PATH = os.getenv("DB_PATH", "/tmp/leaflore.db")


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 0.7,
        updated_at INTEGER NOT NULL,
        UNIQUE(student_id, key)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        student_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at INTEGER NOT NULL
    );
    """)

    conn.commit()
    conn.close()


init_db()

# ----------------------------
# Memory Helpers
# ----------------------------
def set_memory(student_id: str, key: str, value: str):
    conn = _db()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO memories(student_id,key,value,confidence,updated_at)
    VALUES(?,?,?,?,?)
    ON CONFLICT(student_id,key)
    DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at
    """, (student_id, key, value, 0.9, int(time.time())))
    conn.commit()
    conn.close()


def get_memory(student_id: str, key: str) -> Optional[str]:
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT value FROM memories WHERE student_id=? AND key=?",
                (student_id, key))
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else None


def set_stage(student_id: str, stage: str):
    set_memory(student_id, "stage", stage)


def get_stage(student_id: str) -> str:
    return get_memory(student_id, "stage") or "none"


# ----------------------------
# Models
# ----------------------------
class RespondRequest(BaseModel):
    action: str = "respond"  # start_class | respond
    student_input: str = ""
    student_id: Optional[str] = None
    session_id: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
    language: Optional[str] = None


class RespondResponse(BaseModel):
    text: str
    student_id: str
    session_id: str


# ----------------------------
# Utilities
# ----------------------------
def greeting():
    hr = time.localtime().tm_hour
    if 5 <= hr < 12:
        return "Good morning"
    if 12 <= hr < 17:
        return "Good afternoon"
    if 17 <= hr < 22:
        return "Good evening"
    return "Hello"


NAME_RE = re.compile(
    r"^\s*(?:my name is|i am|i'm|im|name is)?\s*([A-Za-z][A-Za-z\s]{1,30})\s*$",
    re.I
)


def normalize_language(text: str) -> Optional[str]:
    t = text.lower().strip()

    if "both" in t:
        return "both"
    if "hindi" in t:
        return "hindi"
    if "english" in t:
        return "english"

    return None


# ----------------------------
# Onboarding Text
# ----------------------------
def intro_text(subject: str):
    return (
        f"{greeting()}! Welcome to Leaflore. "
        f"My name is Anaya. I am your {subject} teacher. "
        "What is your name? "
        "To speak with me, click the Speak button below this screen."
    )


def ask_language(name: str):
    return (
        f"Lovely, {name}. "
        "Which language are you comfortable with — English, Hindi or Both?"
    )


def start_class_text(name: str, lang: str, chapter: str):
    if lang == "english":
        line = "Great! We’ll learn in English."
    elif lang == "hindi":
        line = "बहुत बढ़िया! हम हिंदी में सीखेंगे।"
    else:
        line = "Awesome! We’ll learn in Both (English + Hindi mix)."

    return (
        f"{greeting()}, {name}! {line} "
        f"Today we will learn {chapter}. "
        "It will be a one hour class. "
        "To ask questions, click the Speak button below this screen. "
        "To stop the class, click the Stop button on the top right. "
        "If you stop in between, the class cannot restart from the beginning. "
        "So let’s start learning. Time starts now. "
        f"What do you already know about {chapter}?"
    )


# ----------------------------
# Routes
# ----------------------------
@app.post("/respond", response_model=RespondResponse)
async def respond(req: RespondRequest, request: Request):

    session_id = req.session_id or "default-session"
    student_id = req.student_id or session_id

    subject = req.subject or "your subject"
    chapter = req.chapter or "today’s chapter"

    action = req.action.lower()
    stage = get_stage(student_id)

    # START CLASS
    if action == "start_class":
        set_stage(student_id, "awaiting_name")
        return RespondResponse(
            text=intro_text(subject),
            student_id=student_id,
            session_id=session_id
        )

    student_text = req.student_input.strip()

    # NAME STAGE
    if stage == "awaiting_name":
        m = NAME_RE.search(student_text)
        if m:
            name = m.group(1).strip()
            set_memory(student_id, "student_name", name)
            set_stage(student_id, "awaiting_language")
            return RespondResponse(
                text=ask_language(name),
                student_id=student_id,
                session_id=session_id
            )
        else:
            return RespondResponse(
                text="I didn’t catch your name clearly. Please tell me your name.",
                student_id=student_id,
                session_id=session_id
            )

    # LANGUAGE STAGE
    if stage == "awaiting_language":
        lang = normalize_language(student_text)
        if lang:
            set_memory(student_id, "language", lang)
            set_stage(student_id, "teaching")
            name = get_memory(student_id, "student_name") or "dear student"
            return RespondResponse(
                text=start_class_text(name, lang, chapter),
                student_id=student_id,
                session_id=session_id
            )
        else:
            return RespondResponse(
                text="Please choose one — English, Hindi or Both.",
                student_id=student_id,
                session_id=session_id
            )

    # TEACHING PHASE (simple fallback)
    name = get_memory(student_id, "student_name") or "Student"
    return RespondResponse(
        text=f"{name}, let’s continue learning. Tell me your doubt.",
        student_id=student_id,
        session_id=session_id
    )
