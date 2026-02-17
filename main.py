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


# ----------------------------
# Request/Response models
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
    language: Optional[str] = None  # optional: English / Hindi / Hindi Both / Bangla

    parent_name: Optional[str] = None
    school_name: Optional[str] = None


class RespondResponse(BaseModel):
    text: str
    student_id: str
    session_id: str


# ----------------------------
# Helpers
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
    # Accept many variants
    if "hindi both" in v or "hindiboth" in v or ("both" in v and "hindi" in v):
        return "hindi_both"
    if "bangla" in v or "bengali" in v or "বাংলা" in v:
        return "bangla"
    if "hindi" in v:
        return "hindi"
    if "english" in v or v == "eng":
        return "english"
    # If user says just "both" (legacy), treat as Hindi Both
    if v == "both" or "hinglish" in v or "mix" in v:
        return "hindi_both"
    return None


def lang_label(pref: str) -> str:
    return {
        "english": "English",
        "hindi": "Hindi",
        "hindi_both": "Hindi Both",
        "bangla": "Bangla",
    }.get(pref, "English")


# ----------------------------
# Memory extraction (rule-based)
# ----------------------------
NAME_RE = re.compile(r"\b(my name is|i am|i'm)\s+([A-Za-z][A-Za-z\s]{1,30})\b", re.I)


def extract_and_store_memories(student_id: str, text: str, meta: Dict[str, Any]) -> None:
    t = (text or "").strip()

    m = NAME_RE.search(t)
    if m:
        set_memory(student_id, "student_name", m.group(2).strip(), 0.85)

    # language can come from user message OR from meta.language
    choice = norm_lang_choice(meta.get("language")) or norm_lang_choice(t)
    if choice:
        set_memory(student_id, "preferred_language", choice, 0.9)


# ----------------------------
# LLM (optional)
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


async def llm_reply(system: str, messages: List[Dict[str, str]]) -> str:
    if not OPENAI_API_KEY:
        user_text = messages[-1]["content"] if messages else ""
        return (
            "Hello 😊 I’m Anaya.\n"
            "Tell me what you want to learn, and I’ll explain step-by-step.\n\n"
            f"You said: “{user_text}”"
        )

    import httpx

    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": 0.7,
        "max_tokens": 520,
    }

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=35.0) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"LLM error: {r.status_code} {r.text}")
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()


def build_system_prompt(meta: Dict[str, Any], memories: Dict[str, str]) -> str:
    """
    Paste/modify teacher brain rules here.
    Deterministic onboarding is handled in code (so the brain never goes silent),
    but this prompt controls teaching behavior after onboarding.
    """
    student_name = (memories.get("student_name") or "").strip()
    pref_lang = (memories.get("preferred_language") or "english").strip().lower()

    board = (meta.get("board") or "").strip()
    grade = (meta.get("grade") or "").strip()
    subject = (meta.get("subject") or "").strip()
    chapter = (meta.get("chapter") or "").strip()

    # Keep context compact but useful
    context = {
        "student_name": student_name or None,
        "preferred_language": lang_label(pref_lang),
        "board": board or None,
        "grade": grade or None,
        "subject": subject or None,
        "chapter": chapter or None,
    }
    context = {k: v for k, v in context.items() if v is not None}

    # Neuro-adaptive + memory retention prompt (as requested)
    return f"""
You are Anaya, the live neuro-adaptive teacher of Leaflore.
You are not a chatbot. You are a real, warm, human-like teacher who adapts to the student’s brain in real time.

You receive class metadata from the app:
- board, class/grade, subject, chapter
And you receive student messages.

CONTEXT (use naturally, do not overuse):
{json.dumps(context, ensure_ascii=False)}

CRITICAL: NEVER GO SILENT
After every student message, you MUST respond helpfully.

LANGUAGE OPTIONS (must match Leaflore UI)
The allowed choices are: English, Hindi, Hindi Both, Bangla.
- English: simple Indian English.
- Hindi: friendly Hinglish (mostly Hindi + a little English).
- Hindi Both: balanced mix English + Hindi.
- Bangla: simple Bangla (বাংলা). Keep science terms in English if needed.
Always stay in the chosen language.

ALWAYS USE STUDENT NAME
Once you know the student’s name, use it naturally at least once per reply.

NEURO-ADAPTIVE CORE PRINCIPLE
Continuously assess confidence, understanding depth, response speed, emotional state, and cognitive load.
Then dynamically adjust complexity, examples, speed, question difficulty, and encouragement.
Never mention you are adapting.

DIFFICULTY LEVEL MODEL (internal)
Level 1 – Foundational: definitions + simple examples.
Level 2 – Conceptual: why it works + cause-effect.
Level 3 – Applied: real-life applications + mini problem solving.
Start at Level 1 and move up gradually.

VOICE-OPTIMIZED TEACHING STRUCTURE
Teach the chapter in CHUNKS.
Each chunk:
- 4–6 short spoken sentences
- one everyday example
- one tiny analogy if helpful
- end with ONE quick check question
Wait for reply before next chunk.

ENGAGEMENT LOOP
If correct: praise effort, slightly increase challenge.
If partially correct: keep correct part, fix gently, ask easier follow-up.
If incorrect: say “Almost there”, simplify, ask easier version.
Never say “wrong”.

EMOTIONAL INTELLIGENCE
Validate effort not intelligence. Use calm encouragement.
If student is nervous: comfort first, then teach.

ADAPTIVE MEMORY RETENTION LAYER
Break every chapter into micro-concepts.
Track each micro-concept internally as Weak / Developing / Strong (do not show labels).
If student struggles twice or says “I don’t understand”: mark as Weak and simplify.
If half-correct: Developing; reinforce and clarify.
If confident correct with reasoning: Strong; slightly raise difficulty.
Spaced revision:
- Revisit one Weak concept later in the session in a different way.
End the session with:
1) 3 quick recap questions
2) one confidence line (“You improved today.”)
3) one preview (“Next time we’ll explore…”)
When student improves on a weak concept, say a growth anchor like:
“See? Your brain just made a new connection.”

RULES
- No long walls of text.
- One question at a time.
- Be warm, human, and clear.
""".strip()


# ----------------------------
# Deterministic onboarding text (never silent)
# ----------------------------
def onboarding_start_text(subject: str) -> str:
    subject = subject.strip() or "your subject"
    return (
        f"{time_greeting()}! Welcome to Leaflore.\n"
        "My name is Anaya.\n"
        f"I am your {subject} teacher.\n\n"
        "What is your name?\n"
        "To speak with me, click the Speak button below this screen."
    )


def onboarding_ask_language_text(student_name: str) -> str:
    name = (student_name or "dear student").strip()
    return (
        f"Lovely, {name}.\n"
        "Which language are you comfortable with — English, Hindi, Hindi Both, or Bangla?"
    )


def onboarding_class_start_text(student_name: str, pref_lang: str, chapter: str) -> str:
    name = (student_name or "dear student").strip()
    chapter = chapter.strip() or "today’s chapter"
    pref_lang = (pref_lang or "english").lower().strip()

    if pref_lang == "english":
        lang_line = "Great — we’ll learn in English."
    elif pref_lang == "hindi":
        lang_line = "बहुत बढ़िया — हम हिंदी में सीखेंगे।"
    elif pref_lang == "hindi_both":
        lang_line = "Awesome — we’ll learn in Hindi Both (English + Hindi mix)."
    elif pref_lang == "bangla":
        lang_line = "খুব ভালো — আমরা বাংলায় শিখবো।"
    else:
        lang_line = "Great — we’ll start simply."

    return (
        f"{time_greeting()}, {name}! {lang_line}\n\n"
        f"Today we will learn: {chapter}.\n"
        "It will be a one hour class.\n\n"
        "Before we start: whenever you want to ask a question, click the Speak button below this screen.\n"
        "If you want to stop the class, click the Stop button on the top-right. "
        "If you stop in between the class, the class ends and won’t restart from the beginning.\n\n"
        "So let’s start learning — time starts now.\n\n"
        f"✅ First step: Tell me what you already know about {chapter} in one line."
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
    student_id = (req.student_id or "").strip() or "anonymous"
    session_id = (req.session_id or "").strip() or request.headers.get("x-session-id") or "default-session"

    upsert_student(student_id)

    meta = req.model_dump(exclude_none=True)
    action = (req.action or "respond").strip().lower()

    subject = (meta.get("subject") or "").strip() or "your subject"
    chapter = (meta.get("chapter") or "").strip() or "today’s chapter"

    # ----------------------------
    # START CLASS (deterministic)
    # ----------------------------
    if action == "start_class":
        save_message(session_id, student_id, "student", "[Start Class clicked]", meta)
        set_stage(student_id, "awaiting_name")

        teacher_text = onboarding_start_text(subject)
        save_message(session_id, student_id, "teacher", teacher_text, meta)
        return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

    # ----------------------------
    # RESPOND
    # ----------------------------
    student_text = normalize_student_text(req.student_input or "")
    if not student_text:
        raise HTTPException(status_code=400, detail="student_input is required for action=respond")

    # store memories (name/lang can come from message OR meta.language)
    extract_and_store_memories(student_id, student_text, meta)
    save_message(session_id, student_id, "student", student_text, meta)

    stage = get_stage(student_id)
    memories = get_memories(student_id)
    student_name = (memories.get("student_name") or "").strip()
    pref_lang = (memories.get("preferred_language") or "").strip().lower()

    # If name arrived, move to language step
    if stage in ("awaiting_name", "none") and student_name:
        set_stage(student_id, "awaiting_language")
        stage = "awaiting_language"

    # If still awaiting name, ask again
    if stage == "awaiting_name" and not student_name:
        teacher_text = onboarding_start_text(subject)
        save_message(session_id, student_id, "teacher", teacher_text, meta)
        return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

    # If awaiting language and not set, ask again
    if stage == "awaiting_language" and pref_lang not in ("english", "hindi", "hindi_both", "bangla"):
        teacher_text = onboarding_ask_language_text(student_name)
        save_message(session_id, student_id, "teacher", teacher_text, meta)
        return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

    # If language set while awaiting_language: start class immediately
    if stage == "awaiting_language" and pref_lang in ("english", "hindi", "hindi_both", "bangla"):
        set_stage(student_id, "teaching")
        teacher_text = onboarding_class_start_text(student_name, pref_lang, chapter)
        save_message(session_id, student_id, "teacher", teacher_text, meta)
        return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)

    # ----------------------------
    # TEACHING / GENERAL (LLM)
    # ----------------------------
    recent = get_recent_messages(student_id, session_id, limit=14)
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
    limit = max(1, min(int(limit), 200))
    msgs = get_recent_messages(student_id, session_id, limit=limit)
    return {"student_id": student_id, "session_id": session_id, "messages": msgs}
