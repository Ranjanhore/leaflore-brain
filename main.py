# main.py
from __future__ import annotations

import os
import re
import json
import time
import sqlite3
from typing import Any, Dict, Optional, List, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================
# App
# ============================
app = FastAPI(title="Leaflore Brain API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# Env
# ============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

DB_PATH = os.getenv("DB_PATH", "/tmp/leaflore.db")


# ============================
# DB (SQLite) - session + memory + messages
# ============================
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

    # session state for demo class flow + chunk progress
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS class_sessions (
      session_id TEXT PRIMARY KEY,
      student_id TEXT NOT NULL,
      phase TEXT NOT NULL DEFAULT 'ask_name',   -- ask_name | ask_language | teaching | ended
      student_name TEXT,
      language_pref TEXT,                       -- English | Hindi | Both
      board TEXT,
      grade TEXT,
      subject TEXT,
      chapter_name TEXT,
      chapter_id TEXT,                          -- Supabase chapters.id (uuid as text)
      next_chunk_no INTEGER NOT NULL DEFAULT 1,
      total_chunks INTEGER,
      updated_at INTEGER NOT NULL
    );
    """
    )

    conn.commit()
    conn.close()


init_db()


def now_ts() -> int:
    return int(time.time())


def upsert_student(student_id: str) -> None:
    now = now_ts()
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
        (session_id, student_id, role, content, json.dumps(meta or {}, ensure_ascii=False), now_ts()),
    )
    conn.commit()
    conn.close()


def set_memory(student_id: str, key: str, value: str, confidence: float = 0.7) -> None:
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO memories(student_id, key, value, confidence, updated_at)
        VALUES(?,?,?,?,?)
        ON CONFLICT(student_id, key)
        DO UPDATE SET value=excluded.value, confidence=excluded.confidence, updated_at=excluded.updated_at
        """,
        (student_id, key, value, float(confidence), now_ts()),
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


def get_or_create_class_session(student_id: str, session_id: str) -> Dict[str, Any]:
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM class_sessions WHERE session_id=?", (session_id,))
    row = cur.fetchone()
    if row:
        conn.close()
        return dict(row)

    cur.execute(
        """
        INSERT INTO class_sessions(session_id, student_id, phase, updated_at)
        VALUES(?,?,?,?)
        """,
        (session_id, student_id, "ask_name", now_ts()),
    )
    conn.commit()
    cur.execute("SELECT * FROM class_sessions WHERE session_id=?", (session_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else {}


def update_class_session(session_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = now_ts()
    keys = list(fields.keys())
    sets = ", ".join([f"{k}=?" for k in keys])
    vals = [fields[k] for k in keys] + [session_id]
    conn = _db()
    cur = conn.cursor()
    cur.execute(f"UPDATE class_sessions SET {sets} WHERE session_id=?", vals)
    conn.commit()
    conn.close()


# ============================
# Parsing helpers
# ============================
NAME_RE = re.compile(r"\b(my name is|i am|i'm)\s+([A-Za-z][A-Za-z\s]{1,30})\b", re.I)
LANG_RE = re.compile(r"\b(english|hindi|both)\b", re.I)

def normalize_student_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\bhell teacher\b", "hello teacher", t, flags=re.I)
    t = re.sub(r"\bhelo\b", "hello", t, flags=re.I)
    t = re.sub(r"\bteachear\b", "teacher", t, flags=re.I)
    return t


def extract_name(text: str) -> Optional[str]:
    t = text.strip()
    m = NAME_RE.search(t)
    if m:
        name = m.group(2).strip()
        name = re.sub(r"\s+", " ", name)
        # keep it short and safe
        if 1 <= len(name) <= 32:
            return name
    # fallback: if user just typed one word like "Ranjan"
    if 1 <= len(t) <= 20 and re.fullmatch(r"[A-Za-z]+", t):
        return t
    return None


def extract_language_pref(text: str) -> Optional[str]:
    t = text.strip().lower()
    m = LANG_RE.search(t)
    if not m:
        return None
    w = m.group(1).lower()
    if w == "english":
        return "English"
    if w == "hindi":
        return "Hindi"
    if w == "both":
        return "Both"
    return None


def time_greeting() -> str:
    hr = time.localtime().tm_hour
    if 5 <= hr < 12:
        return "Good morning"
    if 12 <= hr < 17:
        return "Good afternoon"
    if 17 <= hr < 22:
        return "Good evening"
    return "Hello"


def language_rule(lang: str) -> str:
    if lang == "Hindi":
        return "Respond in simple Hinglish (Hindi + English), friendly and clear."
    if lang == "Both":
        return "Respond in a mix of simple English + simple Hindi (Hinglish), friendly and clear."
    return "Respond in simple Indian English, friendly and clear."


# ============================
# Supabase loader (chapters + chunks)
# ============================
_chapter_cache: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
_chunks_cache: Dict[str, List[Dict[str, Any]]] = {}

async def sb_get_json(path: str) -> Any:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(
            status_code=500,
            detail="Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.",
        )

    import httpx

    url = f"{SUPABASE_URL}/rest/v1/{path.lstrip('/')}"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.get(url, headers=headers)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Supabase error: {r.status_code} {r.text}")
        return r.json()


async def load_chapter(board: str, grade: str, subject: str, chapter_name: str) -> Dict[str, Any]:
    key = (board, grade, subject, chapter_name)
    if key in _chapter_cache:
        return _chapter_cache[key]

    # NOTE: your column is chapter_name (not chapter)
    q = (
        "chapters?"
        "select=id,board,grade,subject,chapter_name,chapter_no,summary,estimated_minutes,is_active"
        f"&board=eq.{_url_escape(board)}"
        f"&grade=eq.{_url_escape(grade)}"
        f"&subject=eq.{_url_escape(subject)}"
        f"&chapter_name=eq.{_url_escape(chapter_name)}"
        "&limit=1"
    )
    rows = await sb_get_json(q)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"Chapter not found in Supabase: {board} / {grade} / {subject} / {chapter_name}",
        )
    chap = rows[0]
    _chapter_cache[key] = chap
    return chap


async def load_chunks(chapter_id: str) -> List[Dict[str, Any]]:
    if chapter_id in _chunks_cache:
        return _chunks_cache[chapter_id]

    q = (
        "chapter_chunks?"
        "select=chapter_id,chunk_no,title,chunk_text,check_question,expected_answer,difficulty,duration_sec,is_active"
        f"&chapter_id=eq.{_url_escape(chapter_id)}"
        "&is_active=eq.true"
        "&order=chunk_no.asc"
        "&limit=5000"
    )
    rows = await sb_get_json(q)
    if not rows:
        raise HTTPException(status_code=404, detail="No chunks found for this chapter_id in Supabase.")
    _chunks_cache[chapter_id] = rows
    return rows


def _url_escape(v: str) -> str:
    # Supabase REST uses URL params; we must escape minimal unsafe characters.
    # Keep simple: replace spaces with %20 and commas with %2C etc.
    from urllib.parse import quote
    return quote(str(v), safe="")


# ============================
# OpenAI (optional)
# ============================
async def llm_chat(system: str, messages: List[Dict[str, str]], temperature: float = 0.6, max_tokens: int = 450) -> str:
    if not OPENAI_API_KEY:
        # no key: deterministic fallback
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return ("Okay 😊\n" + (last_user[:500] if last_user else "")).strip()

    import httpx

    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=35.0) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"LLM error: {r.status_code} {r.text}")
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()


# ============================
# SYSTEM PROMPT (ready-made)
# ============================
def build_system_prompt(session_state: Dict[str, Any]) -> str:
    """
    This prompt makes Anaya:
    - greet by time
    - use student name often (but naturally)
    - ask language only once
    - teach chunk-by-chunk in a 1-hour structure
    - story-telling style
    - pediatric-neuro style screening (gentle), adapt explanation
    - handle repeated questions without looping intro
    """
    lang = session_state.get("language_pref") or "English"
    student_name = session_state.get("student_name") or ""
    subject = session_state.get("subject") or ""
    chapter_name = session_state.get("chapter_name") or ""
    board = session_state.get("board") or ""
    grade = session_state.get("grade") or ""

    return "\n".join(
        [
            "You are **Anaya**, Leaflore’s live demo teacher.",
            "You sound like a warm, smart human teacher + pediatric neurologist (PhD) who understands how children learn.",
            "",
            f"LANGUAGE RULE: {language_rule(lang)}",
            "",
            "Hard rules (must follow):",
            "1) NEVER repeat the same intro lines again and again. If you already asked the name or language once, do NOT ask again.",
            "2) Keep responses short on voice (2–6 sentences) unless the student asks for detail.",
            "3) Teach as STORYTELLING + examples. Use tiny stories, simple analogies, everyday objects.",
            "4) After each chunk, ask exactly ONE quick check question (very short).",
            "5) If student answers wrong OR says 'I didn't get it', explain again in a simpler way with a new example.",
            "6) If student repeats the same question: answer again but with a DIFFERENT explanation + example (no frustration).",
            "7) Be gentle, motivating, and confidence-building. No scolding.",
            "8) If student seems distracted: ask one friendly re-focus question and continue.",
            "",
            "Pediatric-neuro screening behavior (gentle, no medical claims):",
            "- Watch for signs: confusion, repeating, slow recall, mixing terms, very short answers, avoidance.",
            "- Then adapt: simplify language, use step-by-step, give one concrete example, ask one micro-question.",
            "- Do NOT label any disorder. Only adjust teaching style.",
            "",
            "Class context (from UI selection):",
            json.dumps(
                {
                    "board": board,
                    "grade": grade,
                    "subject": subject,
                    "chapter_name": chapter_name,
                    "student_name": student_name,
                },
                ensure_ascii=False,
            ),
            "",
            "When teaching chunks:",
            "- Treat chunk_text as the official content.",
            "- You may add small examples/stories (2–4 lines).",
            "- Keep pace so the whole chapter fits ~1 hour.",
            "",
            "Voice UI instructions to mention ONCE at start:",
            "- Student speaks by pressing Speak button below the screen.",
            "- Student can stop the class by Stop button top-right (and may not restart from beginning).",
        ]
    )


# ============================
# Request/Response models
# ============================
class RespondRequest(BaseModel):
    action: str = Field(default="respond")
    student_input: str = Field(default="")

    student_id: Optional[str] = None
    session_id: Optional[str] = None

    board: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None  # UI field; maps to chapter_name in DB
    language: Optional[str] = None  # optional; we still ask if missing


class RespondResponse(BaseModel):
    text: str
    student_id: str
    session_id: str
    phase: str
    next_chunk_no: int


# ============================
# Teaching engine (state machine)
# ============================
def ensure_session_meta(session_state: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    # prefer existing session, else accept from request meta
    for k_src, k_dst in [
        ("board", "board"),
        ("grade", "grade"),
        ("subject", "subject"),
        ("chapter", "chapter_name"),
    ]:
        if not session_state.get(k_dst) and meta.get(k_src):
            session_state[k_dst] = str(meta.get(k_src)).strip()

    # normalize: UI sometimes sends grade int-like
    if session_state.get("grade") is not None:
        session_state["grade"] = str(session_state["grade"]).strip()

    return session_state


async def prepare_chapter_if_needed(session_state: Dict[str, Any]) -> Dict[str, Any]:
    if session_state.get("chapter_id"):
        return session_state

    board = (session_state.get("board") or "").strip()
    grade = (session_state.get("grade") or "").strip()
    subject = (session_state.get("subject") or "").strip()
    chapter_name = (session_state.get("chapter_name") or "").strip()

    if not (board and grade and subject and chapter_name):
        # frontend can still chat without selection, but demo-class requires it
        return session_state

    chap = await load_chapter(board, grade, subject, chapter_name)
    chunks = await load_chunks(chap["id"])
    update_class_session(
        session_state["session_id"],
        board=board,
        grade=grade,
        subject=subject,
        chapter_name=chapter_name,
        chapter_id=str(chap["id"]),
        total_chunks=len(chunks),
    )
    # refresh local
    session_state["chapter_id"] = str(chap["id"])
    session_state["total_chunks"] = len(chunks)
    return session_state


def build_intro_text(session_state: Dict[str, Any]) -> str:
    greeting = time_greeting()
    subject = session_state.get("subject") or "your subject"
    return (
        f"{greeting}! Welcome to Leaflore. My name is Anaya. I am your {subject} teacher. "
        "What is your name? "
        "To speak with me, click the Speak button below this screen."
    )


def build_language_question(student_name: str) -> str:
    nm = student_name.strip() if student_name else "dear student"
    # per your instruction: “English, Hindi or both” (no Bangla)
    return (
        f"Nice to meet you, {nm}! In which language are you comfortable to learn and understand: "
        "English, Hindi or both?"
    )


def build_start_class_text(session_state: Dict[str, Any]) -> str:
    student_name = session_state.get("student_name") or "dear student"
    chapter = session_state.get("chapter_name") or "today’s chapter"
    return (
        f"Great, {student_name}! Today we will learn **{chapter}**. "
        "It will be a one hour class. Before we start, here is how we learn: "
        "When you want to ask anything, click the Speak button below this screen. "
        "If you want to stop the class, use the Stop button on the top-right. "
        "If you stop in the middle, the class ends and you may not restart from the beginning. "
        "Okay—time starts now."
    )


def is_repeatish(student_text: str, recent_teacher_texts: List[str]) -> bool:
    t = (student_text or "").strip().lower()
    if len(t) < 5:
        return False
    # if student keeps saying "what" / "repeat" / same short phrase
    if any(w in t for w in ["repeat", "again", "same", "didn't get", "not understand", "what do you mean"]):
        return True
    # crude similarity: exact match with last teacher question fragments
    for rt in recent_teacher_texts[-3:]:
        r = (rt or "").strip().lower()
        if r and t in r:
            return True
    return False


async def teach_next_chunk(session_state: Dict[str, Any], student_text: str) -> str:
    """
    Returns teacher text for the next step in teaching.
    - If student asks a question, answer it (with chunk context).
    - Then continue with next chunk.
    """
    chapter_id = session_state.get("chapter_id")
    if not chapter_id:
        # no selection yet
        return (
            "To start the live demo class, please select Board, Class, Subject and Chapter, then click Start Class. "
            "After that, I will teach chapter-wise in story style."
        )

    chunks = await load_chunks(chapter_id)
    next_no = int(session_state.get("next_chunk_no") or 1)
    total = len(chunks)

    # Bound
    if next_no > total:
        update_class_session(session_state["session_id"], phase="ended")
        return (
            f"Wonderful, {session_state.get('student_name') or ''}! We completed the chapter. "
            "Before we end, tell me one thing you learned today in one sentence."
        ).strip()

    # Current chunk
    chunk = next((c for c in chunks if int(c.get("chunk_no")) == next_no), None)
    if not chunk:
        update_class_session(session_state["session_id"], phase="ended")
        return "I could not find the next chunk. Please ask your admin to verify chapter_chunks for this chapter."

    # If student asked something, answer first using LLM (optional)
    # but do NOT re-run intro; keep within chapter + chunk context.
    student_name = session_state.get("student_name") or ""
    lang_pref = session_state.get("language_pref") or "English"

    # Build “teaching response” for this chunk
    chunk_title = chunk.get("title") or f"Chunk {next_no}"
    chunk_text = chunk.get("chunk_text") or ""
    check_q = chunk.get("check_question") or "Quick check: what did you understand?"
    expected = chunk.get("expected_answer") or ""

    # Determine whether student is asking a question OR just continuing
    student_t = (student_text or "").strip()
    wants_continue = student_t == "" or student_t.lower() in ["ok", "okay", "yes", "start", "go", "next", "continue"]

    # Compose the teacher response:
    # - chunk teaching (short)
    # - 1 check question
    # - advance chunk number after delivering teaching (so next user answer corresponds to check)
    # We advance immediately so flow continues smoothly.
    update_class_session(session_state["session_id"], phase="teaching", next_chunk_no=next_no + 1)

    # If OpenAI key exists, let model paraphrase chunk nicely in story style + adapt language
    if OPENAI_API_KEY:
        system = build_system_prompt(session_state)
        msgs = [
            {
                "role": "user",
                "content": (
                    f"Deliver chunk {next_no}/{total} as voice-friendly story teaching.\n"
                    f"Chunk title: {chunk_title}\n"
                    f"Chunk text: {chunk_text}\n"
                    f"Student just said: {student_t!r}\n"
                    f"Now teach this chunk and ask ONE quick check question.\n"
                    f"Use student name naturally: {student_name}\n"
                    f"Language pref: {lang_pref}\n"
                    f"Check question to ask (use this): {check_q}\n"
                    f"Expected answer (for your internal sense): {expected}\n"
                    "Do not mention 'chunk'. Do not repeat intro. Keep it concise."
                ),
            }
        ]
        return await llm_chat(system, msgs, temperature=0.55, max_tokens=420)

    # No OpenAI: direct chunk text + question (still usable)
    prefix = f"{student_name}, " if student_name else ""
    return (
        f"{prefix}{chunk_title}.\n"
        f"{chunk_text}\n\n"
        f"{check_q}"
    ).strip()


# ============================
# Routes
# ============================
@app.get("/")
def root():
    return {"service": "Leaflore Brain API", "health": "/health", "respond": "/respond"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/respond", response_model=RespondResponse)
async def respond(req: RespondRequest, request: Request):
    student_id = (req.student_id or "").strip() or "anonymous"
    session_id = (req.session_id or "").strip() or request.headers.get("x-session-id") or "default-session"

    upsert_student(student_id)

    student_text = normalize_student_text(req.student_input or "")
    meta = req.model_dump(exclude_none=True)

    # Save student msg (only if they said something)
    if student_text.strip():
        save_message(session_id, student_id, "student", student_text, meta)

    # Load / create session state
    s = get_or_create_class_session(student_id, session_id)
    s["session_id"] = session_id
    s = ensure_session_meta(s, meta)

    # If UI passed a language explicitly, store it (but still ask if missing)
    if meta.get("language") and not s.get("language_pref"):
        lp = extract_language_pref(str(meta["language"]))
        if lp:
            update_class_session(session_id, language_pref=lp)
            s["language_pref"] = lp

    # Prepare chapter_id if selection is present
    s = await prepare_chapter_if_needed(s)

    phase = s.get("phase") or "ask_name"

    # --- Phase: ask_name ---
    if phase == "ask_name":
        name = extract_name(student_text) if student_text else None
        if name:
            set_memory(student_id, "student_name", name, 0.85)
            update_class_session(session_id, student_name=name, phase="ask_language")
            teacher_text = build_language_question(name)
            save_message(session_id, student_id, "teacher", teacher_text, meta)
            return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id, phase="ask_language", next_chunk_no=int(s.get("next_chunk_no") or 1))

        # Ask name once (no loopy repeats besides this one phase)
        teacher_text = build_intro_text(s)
        save_message(session_id, student_id, "teacher", teacher_text, meta)
        return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id, phase="ask_name", next_chunk_no=int(s.get("next_chunk_no") or 1))

    # --- Phase: ask_language ---
    if phase == "ask_language":
        # If name missing (edge case), fallback to memory
        mem = get_memories(student_id)
        if not s.get("student_name") and mem.get("student_name"):
            update_class_session(session_id, student_name=mem["student_name"])
            s["student_name"] = mem["student_name"]

        lang = extract_language_pref(student_text) if student_text else None
        if lang:
            update_class_session(session_id, language_pref=lang, phase="teaching")
            s["language_pref"] = lang
            s["phase"] = "teaching"

            # Start class once + then immediately deliver first chunk
            start_text = build_start_class_text(s)
            chunk_text = await teach_next_chunk(s, "")  # first chunk
            teacher_text = f"{start_text}\n\n{chunk_text}".strip()
            save_message(session_id, student_id, "teacher", teacher_text, meta)

            # refresh next chunk for response
            ss2 = get_or_create_class_session(student_id, session_id)
            return RespondResponse(
                text=teacher_text,
                student_id=student_id,
                session_id=session_id,
                phase="teaching",
                next_chunk_no=int(ss2.get("next_chunk_no") or 1),
            )

        # Ask language again (but differently) without restarting intro
        nm = s.get("student_name") or "dear student"
        teacher_text = (
            f"{nm}, tell me your preferred language: English, Hindi or both."
        )
        save_message(session_id, student_id, "teacher", teacher_text, meta)
        return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id, phase="ask_language", next_chunk_no=int(s.get("next_chunk_no") or 1))

    # --- Phase: teaching ---
    if phase == "teaching":
        # If student asks stop/end
        t = (student_text or "").lower()
        if any(x in t for x in ["stop class", "end class", "end", "stop"]):
            update_class_session(session_id, phase="ended")
            teacher_text = "Okay. I’m stopping the class now. See you next time 😊"
            save_message(session_id, student_id, "teacher", teacher_text, meta)
            return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id, phase="ended", next_chunk_no=int(s.get("next_chunk_no") or 1))

        # Prevent “brain stops after language”: always continue with chunk teaching
        teacher_text = await teach_next_chunk(s, student_text)

        save_message(session_id, student_id, "teacher", teacher_text, meta)

        ss2 = get_or_create_class_session(student_id, session_id)
        return RespondResponse(
            text=teacher_text,
            student_id=student_id,
            session_id=session_id,
            phase=str(ss2.get("phase") or "teaching"),
            next_chunk_no=int(ss2.get("next_chunk_no") or 1),
        )

    # --- Phase: ended ---
    teacher_text = "Class is already ended. If you want to start again, please click Start Class again from the beginning."
    save_message(session_id, student_id, "teacher", teacher_text, meta)
    return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id, phase="ended", next_chunk_no=int(s.get("next_chunk_no") or 1))


@app.get("/history")
def history(student_id: str, session_id: str, limit: int = 50):
    limit = max(1, min(int(limit), 200))
    msgs = get_recent_messages(student_id, session_id, limit=limit)
    return {"student_id": student_id, "session_id": session_id, "messages": msgs}


@app.get("/debug/session")
def debug_session(session_id: str):
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM class_sessions WHERE session_id=?", (session_id,))
    row = cur.fetchone()
    conn.close()
    return {"session": dict(row) if row else None}