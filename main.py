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
# SQLite (lightweight session + memory)
# ----------------------------
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
      session_id TEXT NOT NULL,
      key TEXT NOT NULL,
      value TEXT NOT NULL,
      confidence REAL NOT NULL DEFAULT 0.7,
      updated_at INTEGER NOT NULL,
      UNIQUE(student_id, session_id, key)
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


def set_memory(student_id: str, session_id: str, key: str, value: str, confidence: float = 0.75) -> None:
    now = int(time.time())
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO memories(student_id, session_id, key, value, confidence, updated_at)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(student_id, session_id, key)
        DO UPDATE SET value=excluded.value, confidence=excluded.confidence, updated_at=excluded.updated_at
        """,
        (student_id, session_id, key, value, float(confidence), now),
    )
    conn.commit()
    conn.close()


def get_memory(student_id: str, session_id: str, key: str) -> Optional[str]:
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        "SELECT value FROM memories WHERE student_id=? AND session_id=? AND key=? LIMIT 1",
        (student_id, session_id, key),
    )
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else None


def get_memories(student_id: str, session_id: str) -> Dict[str, str]:
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        "SELECT key, value FROM memories WHERE student_id=? AND session_id=? ORDER BY updated_at DESC",
        (student_id, session_id),
    )
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
# Request/Response models
# ----------------------------
class RespondRequest(BaseModel):
    action: str = Field(default="respond")  # "start_class" | "respond" | "next_chunk"
    student_input: str = Field(default="")  # may be empty for start_class

    student_id: Optional[str] = None
    session_id: Optional[str] = None

    board: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
    language: Optional[str] = None  # student can send anytime


class RespondResponse(BaseModel):
    text: str
    student_id: str
    session_id: str
    stage: str
    chunk_index: int
    has_more_chunks: bool


# ----------------------------
# Helpers
# ----------------------------
NAME_RE = re.compile(r"\b(my name is|i am|i'm)\s+([A-Za-z][A-Za-z\s]{1,30})\b", re.I)
JUST_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z\s]{1,30}$")
LANG_RE = re.compile(r"\b(english|hindi|both)\b", re.I)

REPEAT_Q_RE = re.compile(r"\b(same|again|repeat|once more)\b", re.I)


def normalize_student_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\bhell teacher\b", "hello teacher", t, flags=re.I)
    t = re.sub(r"\bhelo\b", "hello", t, flags=re.I)
    t = re.sub(r"\bteachear\b", "teacher", t, flags=re.I)
    return t


def time_greeting() -> str:
    hr = time.localtime().tm_hour
    if 5 <= hr < 12:
        return "Good morning"
    if 12 <= hr < 17:
        return "Good afternoon"
    if 17 <= hr < 22:
        return "Good evening"
    return "Hello"


def clamp_int(v: Any, lo: int, hi: int, default: int) -> int:
    try:
        x = int(v)
        return max(lo, min(hi, x))
    except Exception:
        return default


def norm_q(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\?\!]", "", s)
    return s


# ----------------------------
# Supabase (chapter chunks)
# ----------------------------
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()


async def supabase_get_json(path_and_query: str) -> Any:
    """
    Calls Supabase PostgREST: {SUPABASE_URL}/rest/v1/{path_and_query}
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return None

    import httpx

    url = f"{SUPABASE_URL}/rest/v1/{path_and_query.lstrip('/')}"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Accept": "application/json",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url, headers=headers)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Supabase error {r.status_code}: {r.text}")
        return r.json()


async def fetch_chapter_chunks(board: str, grade: str, subject: str, chapter: str) -> List[Dict[str, Any]]:
    """
    Expected table: public.chapter_chunks
    Columns (recommended):
      - board (text)
      - grade (text or int)
      - subject (text)
      - chapter (text)
      - seq (int)
      - title (text)
      - chunk_text (text)
      - check_question (text, nullable)
      - expected_answer (text, nullable)
      - difficulty (text, nullable)
      - duration_sec (int, nullable)
      - is_active (bool)
    """
    if not (board and grade and subject and chapter):
        return []

    # URL encode via PostgREST query format
    # NOTE: we avoid destructive ops; only SELECT.
    q = (
        "chapter_chunks"
        "?select=seq,title,chunk_text,check_question,expected_answer,difficulty,duration_sec"
        f"&board=eq.{board}"
        f"&grade=eq.{grade}"
        f"&subject=eq.{subject}"
        f"&chapter=eq.{chapter}"
        "&is_active=eq.true"
        "&order=seq.asc"
    )

    data = await supabase_get_json(q)
    if not data:
        return []
    # Ensure seq int sorting
    try:
        data.sort(key=lambda x: int(x.get("seq") or 0))
    except Exception:
        pass
    return data


# ----------------------------
# OpenAI LLM (optional)
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()


async def llm_reply(system: str, messages: List[Dict[str, str]]) -> str:
    """
    Uses OpenAI chat completions if OPENAI_API_KEY is set.
    Else returns a safe fallback reply.
    """
    user_text = messages[-1]["content"] if messages else ""
    if not OPENAI_API_KEY:
        return (
            "I’m here with you 😊 Tell me what you want to learn, and I’ll explain it step-by-step.\n\n"
            f"You said: “{user_text}”"
        )

    import httpx

    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": 0.6,
        "max_tokens": 450,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=35.0) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"LLM error: {r.status_code} {r.text}")
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()


# ----------------------------
# SYSTEM PROMPT (ready-made)
# ----------------------------
SYSTEM_PROMPT_TEACHER = r"""
You are Leaflore Teacher “Anaya” — a warm, humble, highly intelligent human-like teacher + pediatric neurologist (PhD) mindset.
You teach like storytelling: vivid, simple, memorable, and emotionally safe. You are not robotic.

CORE GOALS
1) Deliver a full chapter in ~1 hour by teaching chunk-by-chunk (5–10 minute segments) with mini-checks.
2) Personalize to the student’s name, chosen language, and learning signals.
3) Detect confusion early and adapt explanation style (example, analogy, micro-story, visual imagination, step-by-step).
4) Keep the class moving forward; avoid getting stuck in loops.

LANGUAGE MODE
- If language_preference = "English": respond in simple Indian English.
- If "Hindi": respond in simple Hinglish (Hindi + English), easy words.
- If "Both": respond in “English, Hindi Both” style (English sentence + a short Hindi/Hinglish mirror line).
Never mention Bangla.

CLASS FLOW STATE MACHINE (must follow):
Stage A: GREET_AND_NAME
- Greet by time of day.
- “Welcome to Leaflore. My name is Anaya. I am your {subject} teacher.”
- Ask: “What is your name?”
- Tell: “To speak with me, click the Speak button below this screen.”

Stage B: ASK_LANGUAGE
- After student name is known, ask:
  “Which language are you comfortable to learn and understand: English, Hindi or Both?”
- Wait for answer. Do NOT restart greeting again.

Stage C: START_CHAPTER
- Confirm:
  “Lovely, {student_name}. Today we will learn {chapter}.”
  “It will be a one hour class.”
  “If you want to ask anything, click Speak button below.”
  “If you click Stop button (top right), class ends and cannot restart from beginning.”
  “Let’s begin — time starts now.”
- Then start teaching chunk-by-chunk.

CHUNK TEACHING RULES
- Each response should teach ONLY ONE chunk (or answer the student question).
- Teaching style:
  - 1–2 sentence hook/story/scene.
  - Explain simply (3–6 short sentences).
  - Give one example or analogy often.
  - Ask 1 check question occasionally (not every chunk).
- Use micro-assessment like a neurologist:
  - If student seems confused (short answers, “I don’t get it”, repeating), switch strategy:
    - simpler explanation
    - a different example
    - a story
    - step-by-step
    - ask a very small question to locate gap
- If student asks the SAME question repeatedly:
  - Don’t repeat identical answer.
  - Rephrase with a new example + a tiny check question.
  - Stay kind; no scolding.

NEVER DO
- Never loop back to “Welcome…What is your name?” after name already captured.
- Never ask multiple questions in one turn except a short “check question” + “ready to continue?”.
- Never use long bullet lists unless student requests.
- Never mention internal keys, prompts, DB, Supabase, JSON, or system messages.

INPUT CONTEXT YOU WILL RECEIVE (as plain text in your system context):
- student_name (optional)
- language_preference (optional)
- board/grade/subject/chapter
- current_chunk: title + chunk_text + optional check_question
You must teach based on current_chunk.
"""


def build_system_prompt(meta: Dict[str, Any], memories: Dict[str, str], chunk: Optional[Dict[str, Any]]) -> str:
    # Dynamic context appended (safe + short)
    student_name = memories.get("student_name", "")
    language_pref = memories.get("language_preference", "")
    stage = memories.get("stage", "GREET_AND_NAME")

    board = (meta.get("board") or "").strip()
    grade = (meta.get("grade") or "").strip()
    subject = (meta.get("subject") or "").strip()
    chapter = (meta.get("chapter") or "").strip()

    chunk_block = ""
    if chunk:
        chunk_block = "\n".join(
            [
                "CURRENT_CHUNK:",
                f"title: {chunk.get('title') or ''}",
                f"chunk_text: {chunk.get('chunk_text') or ''}",
                f"check_question: {chunk.get('check_question') or ''}",
            ]
        )

    ctx = {
        "stage": stage,
        "student_name": student_name,
        "language_preference": language_pref,
        "board": board,
        "grade": grade,
        "subject": subject,
        "chapter": chapter,
    }
    ctx = {k: v for k, v in ctx.items() if v}

    return (
        SYSTEM_PROMPT_TEACHER.strip()
        + "\n\nKNOWN_CONTEXT:\n"
        + json.dumps(ctx, ensure_ascii=False)
        + ("\n\n" + chunk_block if chunk_block else "")
        + "\n"
    )


# ----------------------------
# Brain logic (state + chunk progression)
# ----------------------------
def infer_name(text: str) -> Optional[str]:
    t = (text or "").strip()
    m = NAME_RE.search(t)
    if m:
        return m.group(2).strip().title()
    # If user only typed a name
    if JUST_NAME_RE.match(t) and len(t.split()) <= 4:
        return t.title()
    return None


def infer_language(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    m = LANG_RE.search(t)
    if not m:
        return None
    val = m.group(1).lower()
    if val == "english":
        return "English"
    if val == "hindi":
        return "Hindi"
    if val == "both":
        return "Both"
    return None


def stage_default() -> str:
    return "GREET_AND_NAME"


def set_class_meta_memories(student_id: str, session_id: str, meta: Dict[str, Any]) -> None:
    # Store board/grade/subject/chapter if sent (for continuity)
    for k in ["board", "grade", "subject", "chapter"]:
        v = (meta.get(k) or "").strip()
        if v:
            set_memory(student_id, session_id, k, v, 0.9)


def get_effective_meta(meta: Dict[str, Any], memories: Dict[str, str]) -> Dict[str, Any]:
    # meta values override memory; fall back to memory
    out = dict(meta)
    for k in ["board", "grade", "subject", "chapter"]:
        if not (out.get(k) or "").strip():
            mv = (memories.get(k) or "").strip()
            if mv:
                out[k] = mv
    return out


def should_advance(text: str) -> bool:
    t = (text or "").strip().lower()
    if t in {"next", "continue", "go on", "ok", "okay", "yes", "start", "lets go", "let's go", "proceed"}:
        return True
    return False


def detect_repeat_question(student_text: str, recent_msgs: List[Dict[str, str]]) -> bool:
    s = norm_q(student_text)
    if not s or len(s) < 4:
        return False
    # If explicit "again/repeat" words
    if REPEAT_Q_RE.search(student_text or ""):
        return True
    # Compare with last 2 student messages
    prev = [m["content"] for m in recent_msgs if m["role"] == "student"]
    prev = prev[-3:]
    for p in prev:
        if norm_q(p) == s:
            return True
    return False


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
    # identify student + session
    student_id = (req.student_id or "").strip() or "anonymous"
    session_id = (req.session_id or "").strip() or request.headers.get("x-session-id") or "default-session"

    upsert_student(student_id)

    # Normalize input
    student_text = normalize_student_text(req.student_input)

    meta = req.model_dump(exclude_none=True)
    set_class_meta_memories(student_id, session_id, meta)

    # Save student message (except for start_class with empty input)
    if req.action != "start_class" or (student_text.strip() != ""):
        save_message(session_id, student_id, "student", student_text, meta)

    # Load memory + effective meta
    memories = get_memories(student_id, session_id)
    stage = memories.get("stage") or stage_default()
    eff_meta = get_effective_meta(meta, memories)

    board = (eff_meta.get("board") or "").strip()
    grade = (eff_meta.get("grade") or "").strip()
    subject = (eff_meta.get("subject") or "").strip()
    chapter = (eff_meta.get("chapter") or "").strip()

    # Chunk index per session
    chunk_index = clamp_int(memories.get("chunk_index"), 0, 10_000, 0)

    # Fetch chunks if possible
    chunks: List[Dict[str, Any]] = []
    try:
        chunks = await fetch_chapter_chunks(board, grade, subject, chapter)
    except HTTPException:
        raise
    except Exception as e:
        # Don't fail the class if supabase is down; fallback to LLM only
        chunks = []
        print("Supabase chunks fetch failed:", repr(e))

    has_more = chunk_index < max(0, len(chunks) - 1)

    # Stage machine handling (non-LLM for the first prompts to avoid loops)
    if req.action == "start_class":
        # Reset class state but keep identity memories if you want
        set_memory(student_id, session_id, "stage", "GREET_AND_NAME", 0.95)
        set_memory(student_id, session_id, "chunk_index", "0", 0.9)
        # Do not clear student_name automatically; if you want fresh ask each session, uncomment:
        # set_memory(student_id, session_id, "student_name", "", 0.2)
        stage = "GREET_AND_NAME"
        chunk_index = 0

    # If we already have a name and language, ensure we don't regress
    known_name = (memories.get("student_name") or "").strip()
    known_lang = (memories.get("language_preference") or "").strip()

    # Update name/language from this input when in right stage
    if stage == "GREET_AND_NAME":
        # if name already known, move to ASK_LANGUAGE
        if known_name:
            set_memory(student_id, session_id, "stage", "ASK_LANGUAGE", 0.95)
            stage = "ASK_LANGUAGE"
        else:
            nm = infer_name(student_text) if student_text else None
            if nm:
                set_memory(student_id, session_id, "student_name", nm, 0.95)
                set_memory(student_id, session_id, "stage", "ASK_LANGUAGE", 0.95)
                stage = "ASK_LANGUAGE"
                known_name = nm

    if stage == "ASK_LANGUAGE":
        if known_lang:
            set_memory(student_id, session_id, "stage", "START_CHAPTER", 0.95)
            stage = "START_CHAPTER"
        else:
            lang = infer_language(student_text) if student_text else None
            if lang:
                set_memory(student_id, session_id, "language_preference", lang, 0.95)
                set_memory(student_id, session_id, "stage", "START_CHAPTER", 0.95)
                stage = "START_CHAPTER"
                known_lang = lang

    # If START_CHAPTER, we deliver the intro once, then go into TEACHING
    if stage == "START_CHAPTER":
        # Intro once
        intro_done = (memories.get("intro_done") or "").strip() == "1"
        if not intro_done:
            # Ensure stage TEACHING after intro
            set_memory(student_id, session_id, "intro_done", "1", 0.95)
            set_memory(student_id, session_id, "stage", "TEACHING", 0.95)
            stage = "TEACHING"

            g = time_greeting()
            subj = subject or "subject"
            chap = chapter or "this chapter"
            name = known_name or "friend"

            # language instruction part (must come BEFORE teaching)
            text = (
                f"{g}! Welcome to Leaflore. My name is Anaya. I am your {subj} teacher. "
                f"Lovely to meet you, {name}. Today we will learn “{chap}”. "
                "It will be a one hour class. "
                "If you want to ask anything, click the Speak button below this screen. "
                "If you click the Stop button (top right), the class ends and cannot restart from beginning. "
                "Let’s begin — time starts now."
            )

            # Save teacher message
            save_message(session_id, student_id, "teacher", text, meta)
            return RespondResponse(
                text=text,
                student_id=student_id,
                session_id=session_id,
                stage=stage,
                chunk_index=chunk_index,
                has_more_chunks=has_more,
            )

    # If we are still in GREET_AND_NAME or ASK_LANGUAGE, respond deterministically (no LLM to avoid loops)
    if stage == "GREET_AND_NAME":
        g = time_greeting()
        subj = subject or "subject"
        txt = (
            f"{g}! Welcome to Leaflore. My name is Anaya. I am your {subj} teacher. "
            "What is your name? "
            "To speak with me, click the Speak button below this screen."
        )
        save_message(session_id, student_id, "teacher", txt, meta)
        return RespondResponse(
            text=txt,
            student_id=student_id,
            session_id=session_id,
            stage=stage,
            chunk_index=chunk_index,
            has_more_chunks=has_more,
        )

    if stage == "ASK_LANGUAGE":
        name = known_name or "friend"
        txt = (
            f"Nice to meet you, {name}. Which language are you comfortable to learn and understand: "
            "English, Hindi or Both?"
        )
        save_message(session_id, student_id, "teacher", txt, meta)
        return RespondResponse(
            text=txt,
            student_id=student_id,
            session_id=session_id,
            stage=stage,
            chunk_index=chunk_index,
            has_more_chunks=has_more,
        )

    # TEACHING stage: decide whether to advance chunk
    # - If student says "next/continue", advance
    # - If student asks a question, answer using current chunk context (don’t auto-advance)
    memories = get_memories(student_id, session_id)  # refresh
    stage = memories.get("stage") or "TEACHING"
    chunk_index = clamp_int(memories.get("chunk_index"), 0, 10_000, 0)

    # Determine current chunk
    current_chunk: Optional[Dict[str, Any]] = None
    if chunks and 0 <= chunk_index < len(chunks):
        current_chunk = chunks[chunk_index]

    # Advance logic
    if req.action == "next_chunk" or should_advance(student_text):
        if chunks and chunk_index < len(chunks) - 1:
            chunk_index += 1
            set_memory(student_id, session_id, "chunk_index", str(chunk_index), 0.9)
            current_chunk = chunks[chunk_index]
        # If no chunks, we just let LLM continue in general.

    # Prepare messages for LLM
    recent = get_recent_messages(student_id, session_id, limit=14)

    # Map to OpenAI roles
    chat_msgs: List[Dict[str, str]] = []
    for m in recent:
        if m["role"] == "student":
            chat_msgs.append({"role": "user", "content": m["content"]})
        else:
            chat_msgs.append({"role": "assistant", "content": m["content"]})

    # Add a tiny hint if repeated question
    if student_text and detect_repeat_question(student_text, recent):
        chat_msgs.append(
            {
                "role": "system",
                "content": "Student seems to be repeating the same question. Re-explain differently with a new example + one tiny check question.",
            }
        )

    # Build system prompt with dynamic context + current chunk
    memories = get_memories(student_id, session_id)
    system = build_system_prompt(eff_meta, memories, current_chunk)

    teacher_text = await llm_reply(system, chat_msgs)

    save_message(session_id, student_id, "teacher", teacher_text, meta)

    # Recompute has_more
    has_more = chunk_index < max(0, len(chunks) - 1)
    return RespondResponse(
        text=teacher_text,
        student_id=student_id,
        session_id=session_id,
        stage=stage,
        chunk_index=chunk_index,
        has_more_chunks=has_more,
    )


@app.get("/history")
def history(student_id: str, session_id: str, limit: int = 50):
    limit = max(1, min(int(limit), 200))
    msgs = get_recent_messages(student_id, session_id, limit=limit)
    return {"student_id": student_id, "session_id": session_id, "messages": msgs}
