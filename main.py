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
# SQLite (light session + memory)
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
        CREATE TABLE IF NOT EXISTS memories (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          student_id TEXT NOT NULL,
          key TEXT NOT NULL,
          value TEXT NOT NULL,
          confidence REAL NOT NULL DEFAULT 0.7,
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


def save_message(session_id: str, student_id: str, role: str, content: str, meta: Optional[dict]) -> None:
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages(session_id, student_id, role, content, meta_json, created_at) VALUES(?,?,?,?,?,?)",
        (session_id, student_id, role, content, json.dumps(meta or {}, ensure_ascii=False), int(time.time())),
    )
    conn.commit()
    conn.close()


def set_memory(student_id: str, key: str, value: str, confidence: float = 0.9) -> None:
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


def get_memory(student_id: str, key: str) -> Optional[str]:
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT value FROM memories WHERE student_id=? AND key=? LIMIT 1", (student_id, key))
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else None


def get_memories(student_id: str) -> Dict[str, str]:
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM memories WHERE student_id=? ORDER BY updated_at DESC", (student_id,))
    rows = cur.fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}


def set_stage(student_id: str, stage: str) -> None:
    set_memory(student_id, "onboarding_stage", stage, 0.95)


def get_stage(student_id: str) -> str:
    return (get_memory(student_id, "onboarding_stage") or "").strip() or "none"


def set_int(student_id: str, key: str, n: int) -> None:
    set_memory(student_id, key, str(int(n)), 0.95)


def get_int(student_id: str, key: str, default: int = 0) -> int:
    v = (get_memory(student_id, key) or "").strip()
    try:
        return int(v)
    except Exception:
        return default


# ----------------------------
# Request/Response models
# ----------------------------
class RespondRequest(BaseModel):
    # actions:
    # - "start_class": teacher speaks first
    # - "respond": student replies / continues class
    action: str = Field(default="respond")
    student_input: str = Field(default="")

    # identifiers (frontend should send stable values each request)
    student_id: Optional[str] = None
    session_id: Optional[str] = None

    # selection before Start Class
    board: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None

    # optional (frontend may send; else teacher asks)
    language: Optional[str] = None  # English / Hindi / Both


class RespondResponse(BaseModel):
    text: str
    student_id: str
    session_id: str


# ----------------------------
# Utilities
# ----------------------------
def time_greeting() -> str:
    hr = time.localtime().tm_hour
    if 5 <= hr < 12:
        return "Good morning"
    if 12 <= hr < 17:
        return "Good afternoon"
    if 17 <= hr < 22:
        return "Good evening"
    return "Hello"


def normalize_student_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\bhell teacher\b", "hello teacher", t, flags=re.I)
    t = re.sub(r"\bhelo\b", "hello", t, flags=re.I)
    t = re.sub(r"\bteachear\b", "teacher", t, flags=re.I)
    return t


def norm_lang_choice(x: Optional[str]) -> Optional[str]:
    if not x:
        return None
    v = x.strip().lower()
    if "both" in v:
        return "both"
    if "hindi" in v:
        return "hindi"
    if "english" in v:
        return "english"
    return None


def lang_label(pref: str) -> str:
    return {"english": "English", "hindi": "Hindi", "both": "Both"}.get(pref, "English")


# ----------------------------
# Name extraction (robust)
# ----------------------------
NAME_RE = re.compile(
    r"^\s*(?:my name is|i am|i'm|im|name is|mera naam|mera naam hai)?\s*([A-Za-z][A-Za-z\s]{1,30})\s*$",
    re.I,
)

_STOP_WORDS = {
    "yes",
    "yeah",
    "yup",
    "ok",
    "okay",
    "hmm",
    "hm",
    "english",
    "hindi",
    "both",
    "hello",
    "hi",
    "start",
    "science",
    "math",
    "maths",
}


def extract_and_store_memories(student_id: str, student_text: str, meta: Dict[str, Any]) -> None:
    t = (student_text or "").strip()

    m = NAME_RE.search(t)
    if m:
        guessed = (m.group(1) or "").strip()
        low = guessed.lower().strip()
        if len(guessed) >= 2 and low not in _STOP_WORDS:
            set_memory(student_id, "student_name", guessed, 0.9)

    choice = norm_lang_choice(meta.get("language")) or norm_lang_choice(t)
    if choice:
        set_memory(student_id, "preferred_language", choice, 0.95)


# ----------------------------
# Supabase (chapter chunks)
# ----------------------------
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()

# In-memory cache (warm) to reduce Supabase calls
# key = "board|grade|subject|chapter" -> (chapter_id, chunks_list)
_CHUNK_CACHE: Dict[str, Tuple[str, List[Dict[str, Any]]]] = {}


class ChunkNotFound(Exception):
    pass


async def supabase_fetch_chapter_and_chunks(
    board: str, grade: str, subject: str, chapter: str
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Fetch chapter + ordered chunks from Supabase tables:
      - chapters(id, board, grade, subject, chapter, is_active)
      - chapter_chunks(chapter_id, seq, chunk_text, check_question, expected_answer, is_active)
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(
            status_code=500,
            detail="Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY on server.",
        )

    key = f"{board}|{grade}|{subject}|{chapter}".lower().strip()
    if key in _CHUNK_CACHE:
        return _CHUNK_CACHE[key]

    import httpx

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }

    # 1) Get chapter id
    chapters_url = f"{SUPABASE_URL}/rest/v1/chapters"
    params = {
        "select": "id,board,grade,subject,chapter,is_active",
        "board": f"eq.{board}",
        "grade": f"eq.{grade}",
        "subject": f"eq.{subject}",
        "chapter": f"eq.{chapter}",
        "is_active": "eq.true",
        "limit": "1",
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(chapters_url, headers=headers, params=params)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Supabase chapters fetch failed: {r.status_code} {r.text}")
        arr = r.json() or []
        if not arr:
            raise ChunkNotFound(f"Chapter not found for {board}/{grade}/{subject}/{chapter}")
        chapter_id = arr[0]["id"]

        # 2) Get ordered chunks
        chunks_url = f"{SUPABASE_URL}/rest/v1/chapter_chunks"
        params2 = {
            "select": "seq,title,chunk_text,check_question,expected_answer,difficulty,is_active",
            "chapter_id": f"eq.{chapter_id}",
            "is_active": "eq.true",
            "order": "seq.asc",
        }
        r2 = await client.get(chunks_url, headers=headers, params=params2)
        if r2.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Supabase chunks fetch failed: {r2.status_code} {r2.text}")
        chunks = r2.json() or []
        if not chunks:
            raise ChunkNotFound(f"No chunks found for chapter_id={chapter_id}")

    # normalize seq and ensure string fields exist
    out_chunks: List[Dict[str, Any]] = []
    for c in chunks:
        try:
            seq = int(c.get("seq") or 0)
        except Exception:
            seq = 0
        if seq <= 0:
            continue
        out_chunks.append(
            {
                "seq": seq,
                "title": (c.get("title") or "").strip(),
                "chunk_text": (c.get("chunk_text") or "").strip(),
                "check_question": (c.get("check_question") or "").strip(),
                "expected_answer": (c.get("expected_answer") or "").strip(),
                "difficulty": (c.get("difficulty") or "").strip(),
            }
        )

    if not out_chunks:
        raise ChunkNotFound(f"Chunks are empty/invalid for chapter_id={chapter_id}")

    _CHUNK_CACHE[key] = (chapter_id, out_chunks)
    return chapter_id, out_chunks


def chapter_key_from_meta(meta: Dict[str, Any]) -> str:
    board = (meta.get("board") or "").strip()
    grade = (meta.get("grade") or "").strip()
    subject = (meta.get("subject") or "").strip()
    chapter = (meta.get("chapter") or "").strip()
    return f"{board}|{grade}|{subject}|{chapter}".lower().strip()


def get_meta_fields(meta: Dict[str, Any]) -> Tuple[str, str, str, str]:
    board = (meta.get("board") or "").strip()
    grade = (meta.get("grade") or "").strip()
    subject = (meta.get("subject") or "").strip()
    chapter = (meta.get("chapter") or "").strip()
    return board, grade, subject, chapter


# ----------------------------
# Teaching text (deterministic onboarding + chunk flow)
# ----------------------------
def onboarding_start_text(subject: str) -> str:
    subject = subject.strip() or "your subject"
    return (
        f"{time_greeting()}! Welcome to Leaflore. "
        "My name is Anaya. "
        f"I am your {subject} teacher. "
        "What is your name? "
        "To speak with me, click the Speak button below this screen."
    )


def onboarding_ask_language_text(student_name: str) -> str:
    name = (student_name or "dear student").strip()
    return f"Lovely, {name}. Which language are you comfortable with — English, Hindi or Both?"


def class_rules_text(student_name: str, lang_pref: str, chapter: str) -> str:
    name = (student_name or "dear student").strip()
    chapter = chapter.strip() or "today’s chapter"
    lang_pref = (lang_pref or "english").strip().lower()

    if lang_pref == "english":
        lang_line = "Great! We’ll learn in English."
    elif lang_pref == "hindi":
        lang_line = "बहुत बढ़िया! हम हिंदी में सीखेंगे।"
    else:
        lang_line = "Awesome! We’ll learn in Both (English + Hindi mix)."

    return (
        f"{time_greeting()}, {name}! {lang_line} "
        f"Today we will learn {chapter}. "
        "It will be a one hour class. "
        "To ask questions, click the Speak button below this screen. "
        "To stop the class, click the Stop button on the top-right. "
        "If you stop in between the class, the class ends and won’t restart from the beginning. "
        "So let’s start learning — time starts now."
    )


def format_chunk(
    student_name: str,
    lang_pref: str,
    subject: str,
    chapter: str,
    chunk: Dict[str, Any],
    is_first: bool,
) -> str:
    """
    Produce the teacher message for a chunk.
    Keep it voice-friendly and short.
    """
    name = (student_name or "dear student").strip()
    lang_pref = (lang_pref or "english").strip().lower()
    subject = subject.strip() or "your subject"
    chapter = chapter.strip() or "this chapter"

    chunk_text = (chunk.get("chunk_text") or "").strip()
    question = (chunk.get("check_question") or "").strip()

    if lang_pref == "hindi":
        # short Hinglish/Hindi wrapper
        prefix = f"{name}, चलिए शुरू करते हैं. "
        if is_first:
            prefix = f"{name}, चलिए {subject} में {chapter} शुरू करते हैं. "
        suffix = f"Quick check: {question}" if question else "Quick check: समझ आया?"
        return f"{prefix}{chunk_text}\n\n{suffix}"

    if lang_pref == "both":
        prefix = f"{name}, let’s start. "
        if is_first:
            prefix = f"{name}, let’s start {subject} — {chapter}. "
        suffix = f"Quick check: {question}" if question else "Quick check: does this make sense?"
        return f"{prefix}{chunk_text}\n\n{suffix}"

    # english default
    prefix = f"{name}, let’s begin. "
    if is_first:
        prefix = f"{name}, let’s begin {subject} — {chapter}. "
    suffix = f"Quick check: {question}" if question else "Quick check: does this make sense?"
    return f"{prefix}{chunk_text}\n\n{suffix}"


def end_of_chapter_text(student_name: str, lang_pref: str, chapter: str) -> str:
    name = (student_name or "dear student").strip()
    chapter = chapter.strip() or "this chapter"
    lang_pref = (lang_pref or "english").strip().lower()

    if lang_pref == "hindi":
        return (
            f"{name}, बहुत बढ़िया! आज हमने {chapter} पूरा किया.\n"
            "अब 3 quick recap:\n"
            "1) सबसे important point क्या था?\n"
            "2) एक example बताओ.\n"
            "3) अगर मैं पूछूं ‘why?’ तो तुम क्या कहोगे?\n\n"
            "You improved today. Next time we’ll go even deeper."
        )

    if lang_pref == "both":
        return (
            f"{name}, awesome! We completed {chapter}.\n"
            "3 quick recap:\n"
            "1) What was the most important point?\n"
            "2) Give me one example.\n"
            "3) Tell me one ‘why’ behind it.\n\n"
            "You improved today. Next time we’ll explore the next topic."
        )

    return (
        f"{name}, well done! We completed {chapter}.\n"
        "3 quick recap:\n"
        "1) What was the most important point?\n"
        "2) Give one example.\n"
        "3) Tell me one ‘why’ behind it.\n\n"
        "You improved today. Next time we’ll explore the next topic."
    )


# ----------------------------
# Optional: LLM (kept minimal; used only for free-form Qs if you enable)
# ----------------------------
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


async def llm_reply(system: str, messages: List[Dict[str, str]]) -> str:
    """
    Optional. If no key, we answer with a simple helpful fallback.
    """
    if not OPENAI_API_KEY:
        user_text = messages[-1]["content"] if messages else ""
        return (
            "I’m here with you 😊\n"
            "Tell me your doubt in one line, and I’ll explain simply.\n\n"
            f"You said: “{user_text}”"
        )

    import httpx

    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": 0.6,
        "max_tokens": 350,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"LLM error: {r.status_code} {r.text}")
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()


def build_system_prompt(meta: Dict[str, Any], memories: Dict[str, str]) -> str:
    """
    Used ONLY when we decide to answer a free-form student doubt during chunk teaching.
    Chunk progression remains deterministic in code.
    """
    student_name = (memories.get("student_name") or "").strip()
    pref_lang = (memories.get("preferred_language") or "english").strip().lower()
    board, grade, subject, chapter = get_meta_fields(meta)
    return f"""
You are Anaya, a warm human teacher.

Student name: {student_name or "unknown"}
Preferred language: {lang_label(pref_lang)}
Board/Class/Subject/Chapter: {board}/{grade}/{subject}/{chapter}

Rules:
- Answer the student's doubt briefly (2–6 short sentences).
- Use the student's name once if known.
- Use the preferred language (English / Hindi / Both).
- After answering, invite them back to the lesson in one line.
""".strip()


def looks_like_a_question(t: str) -> bool:
    s = (t or "").strip().lower()
    if "?" in s:
        return True
    # common doubt phrases
    for k in ["why", "how", "what", "explain", "doubt", "samajh", "समझ", "kyu", "kyon", "kaise"]:
        if k in s:
            return True
    return False


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/respond", response_model=RespondResponse)
async def respond(req: RespondRequest, request: Request):
    # Stable identifiers: do NOT default to "anonymous" or name memory will break.
    session_id = (req.session_id or "").strip() or request.headers.get("x-session-id") or "default-session"
    student_id = (req.student_id or "").strip() or session_id

    meta = req.model_dump(exclude_none=True)
    action = (req.action or "respond").strip().lower()

    board, grade, subject, chapter = get_meta_fields(meta)
    subject = subject or "your subject"
    chapter = chapter or "today’s chapter"

    # START CLASS: teacher speaks first
    if action == "start_class":
        save_message(session_id, student_id, "student", "[Start Class clicked]", meta)
        set_stage(student_id, "awaiting_name")

        teacher_text = onboarding_start_text(subject)
        save_message(session_id, student_id, "teacher", teacher_text, meta)
        return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

    # RESPOND requires input
    student_text = normalize_student_text(req.student_input or "")
    if not student_text:
        raise HTTPException(status_code=400, detail="student_input is required for action=respond")

    # Store name/language memory
    extract_and_store_memories(student_id, student_text, meta)

    # Save student message
    save_message(session_id, student_id, "student", student_text, meta)

    stage = get_stage(student_id)
    memories = get_memories(student_id)
    student_name = (memories.get("student_name") or "").strip()
    pref_lang = (memories.get("preferred_language") or "").strip().lower()

    # If we are awaiting name and a name was captured, move to awaiting language
    if stage in ("awaiting_name", "none") and student_name:
        set_stage(student_id, "awaiting_language")
        stage = "awaiting_language"

    # Still awaiting name -> ask again
    if stage == "awaiting_name" and not student_name:
        teacher_text = "I didn’t catch your name clearly. Please tell me your name."
        save_message(session_id, student_id, "teacher", teacher_text, meta)
        return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

    # Awaiting language -> ask language
    if stage == "awaiting_language" and pref_lang not in ("english", "hindi", "both"):
        teacher_text = onboarding_ask_language_text(student_name)
        save_message(session_id, student_id, "teacher", teacher_text, meta)
        return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

    # If language chosen -> start teaching + load chunks from Supabase
    if stage == "awaiting_language" and pref_lang in ("english", "hindi", "both"):
        # require selection fields for chunk-based teaching
        if not (board and grade and subject and chapter):
            teacher_text = (
                f"{time_greeting()}, {student_name or 'dear student'}! "
                "Before we start, please select Board, Class, Subject and Chapter in the app."
            )
            save_message(session_id, student_id, "teacher", teacher_text, meta)
            return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

        # fetch chunks
        try:
            chapter_id, chunks = await supabase_fetch_chapter_and_chunks(board, grade, subject, chapter)
        except ChunkNotFound:
            teacher_text = (
                f"{time_greeting()}, {student_name or 'dear student'}! "
                f"I can’t find chunks for {board} Class {grade} {subject} — {chapter} yet. "
                "Please pick another chapter or ask admin to add chunks."
            )
            save_message(session_id, student_id, "teacher", teacher_text, meta)
            return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

        # store teaching session pointers
        set_memory(student_id, "chapter_id", chapter_id, 0.95)
        set_memory(student_id, "chapter_key", chapter_key_from_meta(meta), 0.95)
        set_memory(student_id, "chunks_json", json.dumps(chunks, ensure_ascii=False), 0.95)
        set_int(student_id, "chunk_seq", 1)
        set_stage(student_id, "teaching")

        # send class rules + first chunk
        rules = class_rules_text(student_name, pref_lang, chapter)
        first = chunks[0]
        chunk_msg = format_chunk(student_name, pref_lang, subject, chapter, first, is_first=True)
        teacher_text = f"{rules}\n\n{chunk_msg}"
        save_message(session_id, student_id, "teacher", teacher_text, meta)
        return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

    # TEACHING stage: advance chunk-by-chunk
    if stage == "teaching":
        # if student asks a doubt (question-like), optionally answer via LLM, then continue chunk flow
        chunks_json = get_memory(student_id, "chunks_json") or "[]"
        try:
            chunks = json.loads(chunks_json)
            if not isinstance(chunks, list):
                chunks = []
        except Exception:
            chunks = []

        if not chunks:
            # attempt reload from supabase if cache/memory missing
            if not (board and grade and subject and chapter):
                teacher_text = "I’m missing your chapter selection. Please select Board, Class, Subject and Chapter."
                save_message(session_id, student_id, "teacher", teacher_text, meta)
                return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

            try:
                _, chunks = await supabase_fetch_chapter_and_chunks(board, grade, subject, chapter)
            except Exception:
                teacher_text = "I can’t load chapter chunks right now. Please try again."
                save_message(session_id, student_id, "teacher", teacher_text, meta)
                return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

            set_memory(student_id, "chunks_json", json.dumps(chunks, ensure_ascii=False), 0.95)

        # current seq is 1-based index into ordered list
        cur_seq = get_int(student_id, "chunk_seq", 1)

        # Optionally answer a doubt without breaking progression
        if looks_like_a_question(student_text):
            # Answer briefly, then continue with next chunk (do not stall)
            system = build_system_prompt(meta, memories)
            chat = [{"role": "user", "content": student_text}]
            try:
                answer = await llm_reply(system, chat)
            except Exception:
                answer = "Good question. Let me explain simply, and then we’ll continue."

            # next chunk (cur_seq is already sent previously; move forward)
            next_index = cur_seq  # because chunk_seq stored as "next to send" after first send? we set to 1 at start then sent chunk1 immediately -> now next should be 2
            # fix pointer logic:
            # - After sending chunk1 on start, we should set chunk_seq=2. We'll enforce here:
            if cur_seq == 1:
                next_index = 1
            # We'll compute based on "last_sent_seq" memory.
            last_sent = get_int(student_id, "last_sent_seq", 1)
            # If last_sent is not set, assume 1 was last sent.
            if last_sent <= 0:
                last_sent = 1

            next_seq = last_sent + 1

            if next_seq > len(chunks):
                teacher_text = f"{answer}\n\n{end_of_chapter_text(student_name, pref_lang, chapter)}"
                save_message(session_id, student_id, "teacher", teacher_text, meta)
                return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

            chunk = chunks[next_seq - 1]
            chunk_msg = format_chunk(student_name, pref_lang, subject, chapter, chunk, is_first=False)

            # advance pointers
            set_int(student_id, "last_sent_seq", next_seq)
            set_int(student_id, "chunk_seq", next_seq)

            teacher_text = f"{answer}\n\nNow continuing...\n\n{chunk_msg}"
            save_message(session_id, student_id, "teacher", teacher_text, meta)
            return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

        # Normal progression: send next chunk after any student reply
        last_sent = get_int(student_id, "last_sent_seq", 1)
        if last_sent <= 0:
            last_sent = 1

        next_seq = last_sent + 1

        if next_seq > len(chunks):
            teacher_text = end_of_chapter_text(student_name, pref_lang, chapter)
            save_message(session_id, student_id, "teacher", teacher_text, meta)
            return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

        chunk = chunks[next_seq - 1]
        chunk_msg = format_chunk(student_name, pref_lang, subject, chapter, chunk, is_first=False)

        set_int(student_id, "last_sent_seq", next_seq)
        set_int(student_id, "chunk_seq", next_seq)

        teacher_text = chunk_msg
        save_message(session_id, student_id, "teacher", teacher_text, meta)
        return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

    # Fallback: if stage unknown, restart onboarding safely
    set_stage(student_id, "awaiting_name")
    teacher_text = onboarding_start_text(subject)
    save_message(session_id, student_id, "teacher", teacher_text, meta)
    return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)


@app.get("/history")
def history(student_id: str, session_id: str, limit: int = 50):
    """
    Debug endpoint: see recent messages.
    """
    limit = max(1, min(int(limit), 200))
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT role, content, created_at
        FROM messages
        WHERE student_id=? AND session_id=?
        ORDER BY id DESC
        LIMIT ?
        """,
        (student_id, session_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    msgs = [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in rows[::-1]]
    return {"student_id": student_id, "session_id": session_id, "messages": msgs}
