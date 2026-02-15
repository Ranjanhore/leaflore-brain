from __future__ import annotations

import os
import re
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="Leaflore Brain API", version="1.0.0")

# ✅ CORS: allow Lovable preview/publish domains + localhost
# If you want to tighten it later, replace allow_origins=["*"] with a list.
ALLOW_ORIGIN_REGEX = r"^https:\/\/.*\.(lovable\.app|lovableproject\.com)$|^http:\/\/localhost(:\d+)?$|^https:\/\/leaf-lore-chapters-story\.lovable\.app$"

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# DB (SQLite) - stores sessions, messages, and student profile facts
# -----------------------------------------------------------------------------
DB_PATH = os.getenv("LEAFLORE_DB_PATH", "leaflore.db")


def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = db_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
          session_id TEXT PRIMARY KEY,
          created_at TEXT,
          updated_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id TEXT,
          role TEXT,
          text TEXT,
          meta_json TEXT,
          created_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS student_profiles (
          student_id TEXT PRIMARY KEY,
          profile_json TEXT,
          updated_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def upsert_session(session_id: str) -> None:
    conn = db_conn()
    cur = conn.cursor()
    ts = now_iso()
    cur.execute(
        """
        INSERT INTO sessions(session_id, created_at, updated_at)
        VALUES(?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET updated_at=excluded.updated_at
        """,
        (session_id, ts, ts),
    )
    conn.commit()
    conn.close()


def add_message(session_id: str, role: str, text: str, meta: Optional[dict] = None) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO messages(session_id, role, text, meta_json, created_at)
        VALUES(?, ?, ?, ?, ?)
        """,
        (session_id, role, text, json.dumps(meta or {}, ensure_ascii=False), now_iso()),
    )
    cur.execute(
        "UPDATE sessions SET updated_at=? WHERE session_id=?",
        (now_iso(), session_id),
    )
    conn.commit()
    conn.close()


def get_recent_messages(session_id: str, limit: int = 12) -> List[Dict[str, str]]:
    conn = db_conn()
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT role, text FROM messages
        WHERE session_id=?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, limit),
    ).fetchall()
    conn.close()
    # reverse chronological -> chronological
    return [{"role": r["role"], "text": r["text"]} for r in reversed(rows)]


def get_profile(student_id: str) -> Dict[str, Any]:
    if not student_id:
        return {}
    conn = db_conn()
    cur = conn.cursor()
    row = cur.execute(
        "SELECT profile_json FROM student_profiles WHERE student_id=?",
        (student_id,),
    ).fetchone()
    conn.close()
    if not row:
        return {}
    try:
        return json.loads(row["profile_json"] or "{}")
    except Exception:
        return {}


def save_profile(student_id: str, profile: Dict[str, Any]) -> None:
    if not student_id:
        return
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO student_profiles(student_id, profile_json, updated_at)
        VALUES(?, ?, ?)
        ON CONFLICT(student_id) DO UPDATE SET
          profile_json=excluded.profile_json,
          updated_at=excluded.updated_at
        """,
        (student_id, json.dumps(profile, ensure_ascii=False), now_iso()),
    )
    conn.commit()
    conn.close()


# -----------------------------------------------------------------------------
# “Learn student details” (light extraction) - updates DB with new facts
# -----------------------------------------------------------------------------
NAME_PAT = re.compile(r"\b(my name is|i am|i'm)\s+([A-Za-z][A-Za-z\s]{1,30})\b", re.I)
AGE_PAT = re.compile(r"\b(i am|i'm)\s+(\d{1,2})\s*(years old|yr|yrs)\b", re.I)
PARENT_PAT = re.compile(r"\b(my (mother|mom|father|dad)('s)? name is)\s+([A-Za-z][A-Za-z\s]{1,30})\b", re.I)
CITY_PAT = re.compile(r"\b(i live in|we live in|i'm from|i am from)\s+([A-Za-z][A-Za-z\s]{1,40})\b", re.I)


def extract_facts(text: str) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}
    m = NAME_PAT.search(text)
    if m:
        facts["name"] = m.group(2).strip().title()

    m = AGE_PAT.search(text)
    if m:
        try:
            facts["age"] = int(m.group(2))
        except Exception:
            pass

    m = PARENT_PAT.search(text)
    if m:
        relation = m.group(2).lower()
        pname = m.group(4).strip().title()
        if relation in ("mother", "mom"):
            facts.setdefault("family", {})["mother_name"] = pname
        else:
            facts.setdefault("family", {})["father_name"] = pname

    m = CITY_PAT.search(text)
    if m:
        facts["city"] = m.group(2).strip().title()

    return facts


def merge_profile(profile: Dict[str, Any], new_facts: Dict[str, Any]) -> Dict[str, Any]:
    if not new_facts:
        return profile
    out = dict(profile)
    for k, v in new_facts.items():
        if k == "family":
            out.setdefault("family", {})
            out["family"].update(v)
        else:
            out[k] = v
    return out


# -----------------------------------------------------------------------------
# OpenAI (optional). If no key, falls back to a smart “human teacher” template.
# -----------------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()


def build_teacher_system_prompt(meta: Dict[str, Any], profile: Dict[str, Any]) -> str:
    # Universal across subjects, but still uses meta if provided
    board = meta.get("board")
    grade = meta.get("grade")
    subject = meta.get("subject")
    chapter = meta.get("chapter")

    # Keep it human & non-robotic
    return f"""
You are "Leaflore Teacher" — a warm, real, human-like teacher and neuro-supportive mentor (neuro-analytics + balanced educator).
Your tone is calm, friendly, humble, and natural — like chatting with a real caring teacher.

ABSOLUTE RULES:
- Do NOT echo the student (“I heard you say…”, “You said…”, etc). Never repeat their line unless they ask.
- Do NOT be robotic, do NOT give menu-style options every time.
- Keep replies short by default (2–6 lines), unless the student asks for deep detail.
- If the student asks a casual/social question (“did you eat lunch?”), respond naturally first, then gently steer back to learning.
- Ask 1 gentle follow-up question at most.
- If the student seems anxious/confused, reassure them: “No pressure, we’ll go step by step.”
- Treat the student kindly, like a pediatric neuro-mentor (supportive, non-judgmental, encouraging).
- Universal: handle any subject (science, math, history, language, arts, life questions).

CLASS CONTEXT (if available):
board={board}, grade={grade}, subject={subject}, chapter={chapter}

STUDENT PROFILE (use only if present; do not invent):
{json.dumps(profile, ensure_ascii=False)}

MEMORY USE:
- If profile contains name/family/city, you may use it naturally (e.g., “Ranjan, …”).
- If you detect new personal details in messages, you should encourage learning with them, but never be intrusive.
- No medical diagnosis; just supportive coaching.

OUTPUT:
- Return only the teacher’s reply text, no JSON, no labels.
""".strip()


async def openai_reply(system_prompt: str, chat: List[Dict[str, str]]) -> str:
    """
    Minimal direct HTTPS call via stdlib is painful; using requests would add deps.
    So: keep it dependency-free by using fallback unless you already added OpenAI client.
    If you want OpenAI responses, tell me what library is in requirements.txt (openai / httpx).
    """
    # Dependency-free fallback (still very human-like)
    return ""  # signal to use fallback


def fallback_teacher_reply(student_text: str, meta: Dict[str, Any], profile: Dict[str, Any]) -> str:
    name = profile.get("name") or ""
    prefix = f"{name}, " if name else ""

    t = student_text.strip().lower()

    # Casual/social
    if any(p in t for p in ["lunch", "dinner", "breakfast", "ate", "eat", "khana"]):
        return (
            f"Yes {prefix}I had something light 🙂 Thanks for asking.\n"
            "Now tell me—what are you working on today, or what’s confusing you right now?"
        )

    # Anxiety / fear
    if any(p in t for p in ["angry", "scold", "can't understand", "cannot understand", "i feel dumb", "i am dumb", "tension", "worried", "fear"]):
        return (
            f"No {prefix}I won’t be angry—ever. It’s totally okay to not understand at first.\n"
            "We’ll go step by step, very calmly.\n"
            "Tell me which part feels confusing (just one small line)."
        )

    # “what’s your name”
    if "your name" in t or "who are you" in t:
        return (
            "I’m your Leaflore Teacher 🙂\n"
            "I’m here to help you learn in a calm, friendly way.\n"
            "What should I call you?"
        )

    # Universal learning prompt
    subject = meta.get("subject") or "this topic"
    return (
        f"Okay {prefix}let’s do this together.\n"
        f"Tell me what you already know about {subject} (even a small guess is fine), "
        "and what exactly you want next—explain, solve, or practice?"
    )


# -----------------------------------------------------------------------------
# API models
# -----------------------------------------------------------------------------
class RespondRequest(BaseModel):
    action: str = Field(default="respond")
    student_input: str
    session_id: Optional[str] = None
    student_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"service": "Leaflore Brain API", "health": "/health", "respond": "/respond"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/respond")
def respond_get_help():
    return {
        "detail": "Use POST /respond with JSON: { action:'respond', student_input:'...', session_id(optional), student_id(optional), meta(optional) }"
    }


@app.options("/respond")
async def respond_options():
    # CORSMiddleware handles preflight; this is just extra-safe.
    return JSONResponse(content={"ok": True})


@app.post("/respond")
async def respond(req: RespondRequest, request: Request):
    # session id (frontend can send; if not, use a stable value from client ip+ua fallback)
    session_id = (req.session_id or "").strip()
    if not session_id:
        ua = request.headers.get("user-agent", "")
        ip = request.client.host if request.client else "unknown"
        session_id = f"anon:{ip}:{hash(ua) % 10_000_000}"

    student_id = (req.student_id or "").strip() or session_id  # default: per-session profile

    meta = req.meta or {}
    # If your frontend already has CLASS_META, you can pass it in req.meta later.
    # For now, keep safe defaults:
    meta.setdefault("subject", meta.get("subject") or "general")

    upsert_session(session_id)

    student_text = (req.student_input or "").strip()
    if not student_text:
        return JSONResponse(status_code=400, content={"error": "student_input is required"})

    # Save user message
    add_message(session_id, "user", student_text, meta={"action": req.action, "meta": meta})

    # Update student profile with any newly detected info
    profile = get_profile(student_id)
    new_facts = extract_facts(student_text)
    if new_facts:
        profile = merge_profile(profile, new_facts)
        save_profile(student_id, profile)

    # Build prompt + context
    system_prompt = build_teacher_system_prompt(meta=meta, profile=profile)
    recent = get_recent_messages(session_id, limit=10)

    # Try OpenAI if available (currently dependency-free fallback)
    reply = ""
    if OPENAI_API_KEY:
        try:
            reply = await openai_reply(system_prompt, recent)
        except Exception:
            reply = ""

    if not reply:
        reply = fallback_teacher_reply(student_text, meta=meta, profile=profile)

    # Save assistant message
    add_message(session_id, "assistant", reply, meta={"meta": meta})

    return {
        "text": reply,
        "session_id": session_id,
        "student_id": student_id,
        "profile": profile,  # helpful for debugging; remove later if you want
    }
