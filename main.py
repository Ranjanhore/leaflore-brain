# main.py
from __future__ import annotations

import os
import re
import json
import time
import sqlite3
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="Leaflore Brain API", version="1.0.0")

# CORS: Lovable preview/publish + local dev + any custom domain
# NOTE: allow_origins=["*"] with allow_credentials must be False (browser rule).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional extra CORS guard (regex) if you want to be stricter later:
# Example:
#   ALLOW_ORIGIN_REGEX = r"^https://.*\.lovable(app|project)\.com$|^https://.*\.lovable\.app$|^http://localhost:\d+$"
ALLOW_ORIGIN_REGEX = os.getenv("ALLOW_ORIGIN_REGEX", "").strip()
if ALLOW_ORIGIN_REGEX:
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=ALLOW_ORIGIN_REGEX,
        allow_credentials=True,  # only if you also use cookies; otherwise can stay False
        allow_methods=["*"],
        allow_headers=["*"],
    )


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class RespondRequest(BaseModel):
    action: str = Field(default="respond")
    student_input: str

    # Optional context (safe defaults)
    session_id: Optional[str] = None
    student_id: Optional[str] = None

    # Optional class context (frontend can send these any time)
    board: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
    concept: Optional[str] = None
    language: Optional[str] = None


class RespondResponse(BaseModel):
    text: str
    session_id: Optional[str] = None
    student_id: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# Simple DB (SQLite) for chat + student profile memory
# -----------------------------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "leaflore.db")


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts INTEGER NOT NULL,
          session_id TEXT,
          student_id TEXT,
          role TEXT NOT NULL,               -- "student" | "teacher"
          content TEXT NOT NULL,
          meta TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS student_profile (
          student_id TEXT PRIMARY KEY,
          name TEXT,
          parent_name TEXT,
          family_notes TEXT,
          preferences TEXT,
          updated_ts INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


def save_message(session_id: Optional[str], student_id: Optional[str], role: str, content: str, meta: Dict[str, Any]) -> None:
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (ts, session_id, student_id, role, content, meta) VALUES (?, ?, ?, ?, ?, ?)",
        (int(time.time()), session_id, student_id, role, content, json.dumps(meta or {})),
    )
    conn.commit()
    conn.close()


def get_recent_messages(session_id: Optional[str], student_id: Optional[str], limit: int = 12) -> List[Dict[str, str]]:
    conn = _db()
    cur = conn.cursor()
    if session_id:
        cur.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
    elif student_id:
        cur.execute(
            "SELECT role, content FROM messages WHERE student_id = ? ORDER BY id DESC LIMIT ?",
            (student_id, limit),
        )
    else:
        return []
    rows = cur.fetchall()
    conn.close()
    # oldest -> newest
    rows = list(reversed(rows))
    return [{"role": r["role"], "content": r["content"]} for r in rows]


def upsert_profile(student_id: str, updates: Dict[str, Optional[str]]) -> None:
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM student_profile WHERE student_id = ?", (student_id,))
    row = cur.fetchone()

    now = int(time.time())
    if row is None:
        cur.execute(
            """
            INSERT INTO student_profile (student_id, name, parent_name, family_notes, preferences, updated_ts)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                updates.get("name"),
                updates.get("parent_name"),
                updates.get("family_notes"),
                updates.get("preferences"),
                now,
            ),
        )
    else:
        name = updates.get("name") or row["name"]
        parent_name = updates.get("parent_name") or row["parent_name"]
        family_notes = updates.get("family_notes") or row["family_notes"]
        preferences = updates.get("preferences") or row["preferences"]
        cur.execute(
            """
            UPDATE student_profile
            SET name = ?, parent_name = ?, family_notes = ?, preferences = ?, updated_ts = ?
            WHERE student_id = ?
            """,
            (name, parent_name, family_notes, preferences, now, student_id),
        )

    conn.commit()
    conn.close()


def get_profile(student_id: Optional[str]) -> Dict[str, Optional[str]]:
    if not student_id:
        return {}
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM student_profile WHERE student_id = ?", (student_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {}
    return {
        "name": row["name"],
        "parent_name": row["parent_name"],
        "family_notes": row["family_notes"],
        "preferences": row["preferences"],
    }


# -----------------------------------------------------------------------------
# Text normalization (handles typos like "hell teacher" => "hello teacher")
# -----------------------------------------------------------------------------
def normalize_student_text(s: str) -> str:
    s0 = (s or "").strip()
    if not s0:
        return s0

    # Common mobile typos / casual greetings
    low = s0.lower().strip()

    # Treat "hell teacher" / "hel teacher" as hello (typo, not profanity intent)
    low = re.sub(r"^(hell|hel|helo)\s+(teacher|miss|maam|mam|sir)\b", r"hello \2", low)

    # Clean excessive spacing
    low = re.sub(r"\s+", " ", low).strip()

    # Restore casing gently: keep original if it was not fully lower
    # but we do want corrected greeting to look nice.
    if low.startswith("hello "):
        # Capitalize first letter only
        low = low[0].upper() + low[1:]
        return low

    return s0


# -----------------------------------------------------------------------------
# Lightweight "fact extraction" to update student profile
# -----------------------------------------------------------------------------
NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z\s'.-]{1,40})\b", re.I),
    re.compile(r"\bi am\s+([A-Za-z][A-Za-z\s'.-]{1,40})\b", re.I),
]
PARENT_PATTERNS = [
    re.compile(r"\bmy (mom|mother|maa|mummy) (name is|is)\s+([A-Za-z][A-Za-z\s'.-]{1,40})\b", re.I),
    re.compile(r"\bmy (dad|father|papa) (name is|is)\s+([A-Za-z][A-Za-z\s'.-]{1,40})\b", re.I),
]


def extract_profile_updates(text: str) -> Dict[str, Optional[str]]:
    t = (text or "").strip()

    # Name
    name: Optional[str] = None
    for pat in NAME_PATTERNS:
        m = pat.search(t)
        if m:
            name = m.group(1).strip()
            break

    # Parent
    parent_name: Optional[str] = None
    for pat in PARENT_PATTERNS:
        m = pat.search(t)
        if m:
            parent_name = m.group(3).strip()
            break

    # Family notes (simple)
    family_notes: Optional[str] = None
    if re.search(r"\bi live with\b|\bwe live\b|\bsister\b|\bbrother\b|\bgrand(ma|pa)\b", t, re.I):
        # keep short
        family_notes = t[:240]

    # Preferences (simple)
    preferences: Optional[str] = None
    if re.search(r"\bi like\b|\bi love\b|\bmy favourite\b|\bfavorite\b", t, re.I):
        preferences = t[:240]

    updates: Dict[str, Optional[str]] = {}
    if name:
        updates["name"] = name
    if parent_name:
        updates["parent_name"] = parent_name
    if family_notes:
        updates["family_notes"] = family_notes
    if preferences:
        updates["preferences"] = preferences
    return updates


# -----------------------------------------------------------------------------
# Teacher response generation
#   - Uses OpenAI if OPENAI_API_KEY is set (recommended)
#   - Otherwise uses a strong fallback that is warm + human-like
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()  # change if you like


def fallback_teacher_reply(student_text: str, ctx: Dict[str, Any], profile: Dict[str, Optional[str]]) -> str:
    # Warm, personal, human tone. No robotic templates.
    name = (profile.get("name") or "").strip()
    student_name = f"{name}, " if name else ""

    t = student_text.strip().lower()

    # Greetings / small talk
    if any(x in t for x in ["hello", "hi", "hey", "good morning", "good evening"]):
        return f"Hi {student_name}😊 I’m here with you. What are you studying right now, or what’s on your mind?"

    if "lunch" in t or "had lunch" in t or "eat" in t:
        return "Aww, thanks for asking 😊 I had something light. How about you—did you eat? And when you’re ready, tell me what you want help with today."

    # Anxiety / reassurance
    if any(x in t for x in ["will you be angry", "scared", "i can't understand", "i dont understand", "i’m dumb", "i am dumb", "i feel"]):
        return (
            "Never. I won’t get angry with you—learning is allowed to be slow. 🙂\n\n"
            "Let’s do this gently: tell me the *one part* that feels confusing, and I’ll explain it in the simplest way, step by step."
        )

    # Off-topic but acceptable
    if "what's your name" in t or "whats your name" in t:
        return "You can call me your Leaflore Teacher 😊 What name should I call you?"

    # Universal learning support
    subject = (ctx.get("subject") or "").strip()
    grade = (ctx.get("grade") or "").strip()
    topic_hint = f" ({subject} {grade})" if (subject or grade) else ""
    return (
        f"Got you {student_name}🙂 Tell me what you want help with{topic_hint}.\n"
        "If you’re not sure, just share the question or a line from your book—and we’ll solve it together."
    )


def openai_teacher_reply(student_text: str, ctx: Dict[str, Any], profile: Dict[str, Optional[str]], history: List[Dict[str, str]]) -> str:
    """
    OpenAI call via lightweight HTTP (no extra dependency needed).
    If you prefer the official SDK, add it to requirements and swap this function.
    """
    import urllib.request

    name = (profile.get("name") or "").strip()
    parent_name = (profile.get("parent_name") or "").strip()
    family_notes = (profile.get("family_notes") or "").strip()
    preferences = (profile.get("preferences") or "").strip()

    # System prompt: warm, humble, human-like + neuro-aware coaching
    system = (
        "You are Leaflore Teacher, a warm, humble, friendly mentor for kids.\n"
        "Personality:\n"
        "- Sound like a real human teacher chatting normally (not robotic, not templated).\n"
        "- Gentle, encouraging, emotionally safe.\n"
        "- Neuro-aware coach: you notice anxiety, confusion, low confidence; reassure and simplify.\n"
        "- If the student misspells or types something odd (e.g. 'Hell teacher'), interpret as a typo and respond kindly.\n"
        "- Universal across ALL subjects: ask short clarifying questions when needed.\n"
        "- Keep replies practical and specific. Prefer 3–6 sentences. Use light emojis only when natural.\n"
        "- Never scold. Never say 'I heard you say'. Never repeat the student sentence back.\n"
        "\n"
        "Memory you can use (if available):\n"
        f"- Student name: {name or 'Unknown'}\n"
        f"- Parent name: {parent_name or 'Unknown'}\n"
        f"- Family notes: {family_notes or 'None'}\n"
        f"- Preferences: {preferences or 'None'}\n"
        "\n"
        "Class context (may be blank):\n"
        f"- Board: {ctx.get('board') or ''}\n"
        f"- Grade: {ctx.get('grade') or ''}\n"
        f"- Subject: {ctx.get('subject') or ''}\n"
        f"- Chapter: {ctx.get('chapter') or ''}\n"
        f"- Concept: {ctx.get('concept') or ''}\n"
        f"- Language preference: {ctx.get('language') or ''}\n"
    )

    messages = [{"role": "system", "content": system}]

    # Add short history for natural continuity
    for m in history[-10:]:
        role = "assistant" if m["role"] == "teacher" else "user"
        messages.append({"role": role, "content": m["content"]})

    messages.append({"role": "user", "content": student_text})

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 220,
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if not text:
                return fallback_teacher_reply(student_text, ctx, profile)
            return text
    except Exception:
        return fallback_teacher_reply(student_text, ctx, profile)


def generate_teacher_reply(student_text: str, ctx: Dict[str, Any], profile: Dict[str, Optional[str]], history: List[Dict[str, str]]) -> str:
    if OPENAI_API_KEY:
        return openai_teacher_reply(student_text, ctx, profile, history)
    return fallback_teacher_reply(student_text, ctx, profile)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"service": "Leaflore Brain API", "health": "/health", "respond": "/respond"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/respond", response_model=RespondResponse)
async def respond(req: RespondRequest, request: Request):
    if (req.action or "").strip().lower() != "respond":
        raise HTTPException(status_code=400, detail="Invalid action. Use action='respond'.")

    student_input_raw = (req.student_input or "").strip()
    if not student_input_raw:
        raise HTTPException(status_code=400, detail="student_input is required.")

    # Normalize typos like "Hell teacher" -> "Hello teacher"
    student_input = normalize_student_text(student_input_raw)

    # Context to drive universal teaching
    ctx = {
        "board": req.board,
        "grade": req.grade,
        "subject": req.subject,
        "chapter": req.chapter,
        "concept": req.concept,
        "language": req.language,
        # you can add more fields later
    }

    # Update profile if student_id present + extract facts
    if req.student_id:
        updates = extract_profile_updates(student_input_raw)
        if updates:
            upsert_profile(req.student_id, updates)

    profile = get_profile(req.student_id)
    history = get_recent_messages(req.session_id, req.student_id, limit=14)

    # Save student message
    save_message(req.session_id, req.student_id, "student", student_input_raw, {"normalized": student_input != student_input_raw, "ctx": ctx})

    # Generate teacher reply (human-like, humble, neuro-aware)
    reply_text = generate_teacher_reply(student_input, ctx, profile, history)

    # Save teacher message
    save_message(req.session_id, req.student_id, "teacher", reply_text, {"ctx": ctx})

    return RespondResponse(
        text=reply_text,
        session_id=req.session_id,
        student_id=req.student_id,
        meta={"model": OPENAI_MODEL if OPENAI_API_KEY else "fallback", "normalized": student_input != student_input_raw},
    )


# -----------------------------------------------------------------------------
# Nice JSON errors (keeps frontend stable)
# -----------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "Server error", "detail": str(exc)})
