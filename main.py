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
# DB (SQLite) - demo stable
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


def set_memory(student_id: str, key: str, value: str, confidence: float = 0.8) -> None:
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


def get_recent_messages(student_id: str, session_id: str, limit: int = 20) -> List[Dict[str, str]]:
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
    action: str = Field(default="respond")  # "start_class" | "respond"
    student_input: str = Field(default="")

    student_id: Optional[str] = None
    session_id: Optional[str] = None

    board: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
    concept: Optional[str] = None
    language: Optional[str] = None  # optional pre-selected

    parent_name: Optional[str] = None
    school_name: Optional[str] = None


class RespondResponse(BaseModel):
    text: str
    student_id: str
    session_id: str


# ----------------------------
# Utilities
# ----------------------------
NAME_RE = re.compile(r"\b(my name is|i am|i'm)\s+([A-Za-z][A-Za-z\s]{1,30})\b", re.I)


def normalize_student_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\bhell teacher\b", "hello teacher", t, flags=re.I)
    t = re.sub(r"\bhelo\b", "hello", t, flags=re.I)
    t = re.sub(r"\bteachear\b", "teacher", t, flags=re.I)
    return t


def time_greeting() -> str:
    h = time.localtime().tm_hour
    if 5 <= h < 12:
        return "Good morning"
    if 12 <= h < 17:
        return "Good afternoon"
    if 17 <= h < 22:
        return "Good evening"
    return "Hello"


def pick_lang_mode(raw: str) -> str:
    """Returns: EN | HI | BOTH"""
    s = (raw or "").strip().lower()
    if "hindi" in s or s in {"hi", "h"}:
        return "HI"
    if "both" in s or "hinglish" in s or ("english" in s and "hindi" in s):
        return "BOTH"
    if "english" in s or s in {"en", "e"}:
        return "EN"
    return ""


def say(text_en: str, text_hi: str, lang_mode: str) -> str:
    """Option 2: BOTH = Hinglish full time (Hindi+English mixed)."""
    if lang_mode == "HI":
        return text_hi
    if lang_mode == "BOTH":
        # mix nicely
        return f"{text_en} {text_hi}"
    return text_en


# ----------------------------
# Demo chunks (Phase 1: perfect one class first)
# Later we move this to Supabase.
# ----------------------------
DEMO_CHUNKS = [
    {
        "title": "Leaf parts (Lamina, Midrib, Veins, Petiole)",
        "teach_en": (
            "A leaf is like the plant’s kitchen. The flat green part is the lamina. "
            "The strong middle line is the midrib, and small lines are veins that carry water and food. "
            "The petiole is the stalk that connects the leaf to the stem."
        ),
        "teach_hi": (
            "Leaf plant ka kitchen hota hai. Flat green part ko lamina kehte hain. "
            "Beech ki strong line midrib hoti hai, aur chhoti lines veins hoti hain jo pani aur food le jaati hain. "
            "Petiole wo dandi hai jo leaf ko stem se jodti hai."
        ),
        "check_q": "Can you tell me: which part is the middle strong line called?",
        "check_q_hi": "Batao: leaf ki beech wali strong line ko kya kehte hain?",
        "expected_keywords": ["midrib"],
    },
    {
        "title": "Types of leaves (Simple vs Compound)",
        "teach_en": (
            "Some leaves have one single blade—these are simple leaves (like mango). "
            "Some leaves are divided into many small leaflets—these are compound leaves (like neem)."
        ),
        "teach_hi": (
            "Kuch leaves ek hi blade wali hoti hain—simple leaf (jaise aam). "
            "Kuch leaves bahut saare leaflets mein divided hoti hain—compound leaf (jaise neem)."
        ),
        "check_q": "Neem is a simple leaf or compound leaf?",
        "check_q_hi": "Neem simple leaf hai ya compound leaf?",
        "expected_keywords": ["compound"],
    },
    {
        "title": "Venation (Reticulate vs Parallel)",
        "teach_en": (
            "Venation means the pattern of veins. In many dicots like guava, veins make a net—reticulate venation. "
            "In monocots like grass, veins run side by side—parallel venation."
        ),
        "teach_hi": (
            "Venation matlab veins ka pattern. Guava jaise dicots mein veins net banati hain—reticulate venation. "
            "Grass jaise monocots mein veins side-by-side chalti hain—parallel venation."
        ),
        "check_q": "Grass usually has reticulate or parallel venation?",
        "check_q_hi": "Grass mein reticulate hota hai ya parallel venation?",
        "expected_keywords": ["parallel"],
    },
    {
        "title": "Leaf modifications (Spines, Tendrils, Traps)",
        "teach_en": (
            "Leaves can change to help plants survive. Cactus leaves become spines to save water. "
            "Some plants have tendrils for climbing. Venus flytrap has trap-like leaves to catch insects."
        ),
        "teach_hi": (
            "Leaves survival ke liye change ho sakti hain. Cactus ki leaves spines ban jaati hain taaki pani bache. "
            "Kuch plants mein tendrils hoti hain climbing ke liye. Venus flytrap ki leaves trap jaise hoti hain insects pakadne ke liye."
        ),
        "check_q": "Why do cactus leaves become spines?",
        "check_q_hi": "Cactus ki leaves spines kyun ban jaati hain?",
        "expected_keywords": ["save", "water", "reduce", "loss"],
    },
    {
        "title": "Quick recap + confidence check",
        "teach_en": (
            "Quick recap: lamina is the blade, midrib is the main line, veins carry water/food, petiole connects leaf to stem. "
            "Simple vs compound, and venation patterns. Great job so far!"
        ),
        "teach_hi": (
            "Quick recap: lamina blade hai, midrib main line hai, veins pani/food le jaati hain, petiole leaf ko stem se jodta hai. "
            "Simple vs compound, aur venation patterns. Bahut badiya!"
        ),
        "check_q": "One last thing: what is the stalk of a leaf called?",
        "check_q_hi": "Last question: leaf ki dandi ko kya kehte hain?",
        "expected_keywords": ["petiole"],
    },
]


# ----------------------------
# Optional LLM (kept optional)
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


async def llm_reply(system: str, messages: List[Dict[str, str]]) -> str:
    if not OPENAI_API_KEY:
        # fallback if no key (works reliably)
        last = messages[-1]["content"] if messages else ""
        return f"Okay. {last}".strip()

    import httpx

    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": system}] + messages,
        "temperature": 0.6,
        "max_tokens": 260,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"LLM error: {r.status_code} {r.text}")
        data = r.json()
        return (data["choices"][0]["message"]["content"] or "").strip()


def build_demo_system_prompt(meta: Dict[str, Any], memories: Dict[str, str]) -> str:
    # Hard rules to avoid repetition and to follow the state machine
    lang_mode = memories.get("preferred_language_mode", "EN")
    student_name = memories.get("student_name", "")
    greeted = memories.get("demo_greeted", "false") == "true"
    step = memories.get("demo_step", "ASK_NAME")
    subj = meta.get("subject") or memories.get("selected_subject") or "Science"
    chap = meta.get("chapter") or memories.get("selected_chapter") or "The Leaf"

    return "\n".join(
        [
            "You are Anaya, a warm human-like teacher for kids. Keep it short, friendly, and confident.",
            "CRITICAL: Never repeat the full greeting again if demo_greeted=true.",
            "CRITICAL: Follow the step exactly and move forward. Do not get stuck.",
            "CRITICAL: Always address the student by name once you know it.",
            "If preferred_language_mode=HI: reply in Hindi.",
            "If preferred_language_mode=EN: reply in English.",
            "If preferred_language_mode=BOTH: reply in Hinglish (Hindi+English mixed).",
            "",
            f"Current step: {step}",
            f"demo_greeted: {str(greeted).lower()}",
            f"Student name: {student_name}",
            f"Selected subject: {subj}",
            f"Selected chapter: {chap}",
            "",
            "Behavior rules:",
            "- Ask only ONE question at a time.",
            "- If student asks same question repeatedly: answer patiently with a new example or micro-story.",
            "- While teaching: explain one chunk at a time, then ask one quick check question.",
        ]
    )


# ----------------------------
# State Machine Logic
# ----------------------------
def get_step(mem: Dict[str, str]) -> str:
    return mem.get("demo_step") or "ASK_NAME"


def set_step(student_id: str, step: str) -> None:
    set_memory(student_id, "demo_step", step, 0.95)


def set_selected_meta(student_id: str, board: str, grade: str, subject: str, chapter: str) -> None:
    if board:
        set_memory(student_id, "selected_board", board, 0.95)
    if grade:
        set_memory(student_id, "selected_grade", grade, 0.95)
    if subject:
        set_memory(student_id, "selected_subject", subject, 0.95)
    if chapter:
        set_memory(student_id, "selected_chapter", chapter, 0.95)


def try_extract_name(text: str) -> str:
    t = text.strip()
    m = NAME_RE.search(t)
    if m:
        return m.group(2).strip().title()
    # simple fallback: single word name
    words = re.findall(r"[A-Za-z]{2,20}", t)
    if words:
        # avoid "English"/"Hindi"/"Both" being treated as name
        bad = {"english", "hindi", "both", "science", "teacher"}
        if words[0].lower() not in bad:
            return words[0].title()
    return ""


def is_answer_correct(answer: str, expected_keywords: List[str]) -> bool:
    a = (answer or "").strip().lower()
    for k in expected_keywords:
        if k.lower() in a:
            return True
    return False


def teacher_opening(mem: Dict[str, str], subject: str) -> str:
    lang_mode = mem.get("preferred_language_mode", "EN")
    greet = time_greeting()
    base_en = f"{greet}! Welcome to Leaflore. My name is Anaya. I am your {subject} teacher."
    base_hi = f"{greet}! Leaflore mein aapka swagat hai. Mera naam Anaya hai. Main aapki {subject} teacher hoon."
    ask_en = "What is your name? To speak with me, click the Speak button below this screen."
    ask_hi = "Aapka naam kya hai? Mujhse baat karne ke liye neeche Speak button dabaiye."
    return say(base_en, base_hi, lang_mode) + " " + say(ask_en, ask_hi, lang_mode)


def teacher_ask_language(mem: Dict[str, str]) -> str:
    lang_mode = mem.get("preferred_language_mode", "EN")
    en = "Nice to meet you! Which language are you comfortable to learn in: English, Hindi or Both?"
    hi = "Aapse milkar accha laga! Aap kis language mein comfortable ho: English, Hindi ya Both?"
    return say(en, hi, lang_mode)


def teacher_class_rules(mem: Dict[str, str], student_name: str, subject: str, chapter: str) -> str:
    lang_mode = mem.get("preferred_language_mode", "EN")
    en = (
        f"Great, {student_name}! Today we will learn “{chapter}” in {subject}. "
        "This is a one-hour class. Before we start, quick guide: "
        "To ask any question, click the Speak button below. "
        "To stop the class anytime, click Stop at the top-right. "
        "But remember: if you stop in between, I will stop teaching and you can’t restart from the beginning in this demo. "
        "Alright—time starts now. Let’s begin!"
    )
    hi = (
        f"Bahut badiya, {student_name}! Aaj hum “{chapter}” {subject} mein seekhenge. "
        "Ye one-hour class hai. Start karne se pehle ek chhota guide: "
        "Koi bhi question poochna ho to neeche Speak button dabaiye. "
        "Class rokni ho to top-right pe Stop button dabaiye. "
        "Par yaad rahe: demo mein agar aap beech mein stop karoge, to main teaching band kar dungi aur beginning se restart nahi hoga. "
        "Chaliye—time ab start. Shuru karte hain!"
    )
    return say(en, hi, lang_mode)


def teacher_teach_chunk(mem: Dict[str, str], student_name: str, chunk_index: int) -> str:
    lang_mode = mem.get("preferred_language_mode", "EN")
    chunk = DEMO_CHUNKS[min(chunk_index, len(DEMO_CHUNKS) - 1)]
    teach = say(chunk["teach_en"], chunk["teach_hi"], lang_mode)
    q = say(chunk["check_q"], chunk["check_q_hi"], lang_mode)
    # Storytelling style micro-hook
    hook_en = f"Okay {student_name}, imagine you are a tiny explorer inside a leaf."
    hook_hi = f"Okay {student_name}, socho tum ek tiny explorer ho leaf ke andar."
    hook = say(hook_en, hook_hi, lang_mode)
    return f"{hook} {teach}\n\n{q}"


def teacher_remediate(mem: Dict[str, str], student_name: str, chunk_index: int) -> str:
    lang_mode = mem.get("preferred_language_mode", "EN")
    chunk = DEMO_CHUNKS[min(chunk_index, len(DEMO_CHUNKS) - 1)]
    # alternate explanation
    en = (
        f"No worries, {student_name} — let me explain in a simpler way. "
        f"{chunk['title']} is like a road system: the midrib is the main highway, and veins are small streets. "
        "Now try again—"
    )
    hi = (
        f"Koi baat nahi, {student_name} — main aur simple bana deti hoon. "
        f"{chunk['title']} ek road system jaisa hai: midrib main highway hai, aur veins chhoti streets hain. "
        "Ab phir se try karo—"
    )
    q = say(chunk["check_q"], chunk["check_q_hi"], lang_mode)
    return say(en, hi, lang_mode) + "\n\n" + q


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
    # IDs
    student_id = (req.student_id or "").strip() or "anonymous"
    session_id = (req.session_id or "").strip() or request.headers.get("x-session-id") or "default-session"

    upsert_student(student_id)

    meta = req.model_dump(exclude_none=True)
    student_text = normalize_student_text(req.student_input)

    # Load memories
    mem = get_memories(student_id)

    # Store selected dropdowns (board/class/subject/chapter)
    board = (req.board or mem.get("selected_board") or "").strip()
    grade = (req.grade or mem.get("selected_grade") or "").strip()
    subject = (req.subject or mem.get("selected_subject") or "Science").strip()
    chapter = (req.chapter or mem.get("selected_chapter") or "The Leaf").strip()
    set_selected_meta(student_id, board, grade, subject, chapter)

    # Save student message (except __START__ to keep history clean)
    if student_text and student_text != "__START__":
        save_message(session_id, student_id, "student", student_text, meta)

    # Determine step
    step = get_step(mem)
    lang_mode = mem.get("preferred_language_mode", "")

    # ACTION: start_class sets the flow to greeting/ask-name
    if (req.action or "").strip() == "start_class" or student_text == "__START__":
        set_memory(student_id, "demo_greeted", "true", 0.95)
        set_step(student_id, "ASK_NAME")
        set_memory(student_id, "chunk_index", "0", 0.95)
        # do not force language yet; ask after name
        text = teacher_opening(mem, subject)
        save_message(session_id, student_id, "teacher", text, meta)
        return RespondResponse(text=text, student_id=student_id, session_id=session_id)

    # Refresh mem after potential updates
    mem = get_memories(student_id)
    step = get_step(mem)

    # STEP: ASK_NAME
    if step == "ASK_NAME":
        name = try_extract_name(student_text)
        if not name:
            # ask again (short)
            msg = say(
                "Tell me your name please. Then we’ll start.",
                "Apna naam batao please. Phir hum start karenge.",
                mem.get("preferred_language_mode", "EN"),
            )
            save_message(session_id, student_id, "teacher", msg, meta)
            return RespondResponse(text=msg, student_id=student_id, session_id=session_id)

        set_memory(student_id, "student_name", name, 0.95)
        set_step(student_id, "ASK_LANGUAGE")
        # ask language (English, Hindi or Both)
        msg = (
            say(f"Nice to meet you, {name}!", f"Nice to meet you, {name}!", "EN")
            + " "
            + teacher_ask_language(mem)
        )
        save_message(session_id, student_id, "teacher", msg, meta)
        return RespondResponse(text=msg, student_id=student_id, session_id=session_id)

    # STEP: ASK_LANGUAGE
    if step == "ASK_LANGUAGE":
        chosen = pick_lang_mode(student_text)
        if not chosen:
            msg = "Please choose one: English, Hindi or Both."
            save_message(session_id, student_id, "teacher", msg, meta)
            return RespondResponse(text=msg, student_id=student_id, session_id=session_id)

        # Save language mode
        set_memory(student_id, "preferred_language_mode", chosen, 0.95)

        mem = get_memories(student_id)
        name = mem.get("student_name", "friend")
        set_step(student_id, "START_CHAPTER")

        msg = teacher_class_rules(mem, name, subject, chapter)
        save_message(session_id, student_id, "teacher", msg, meta)
        return RespondResponse(text=msg, student_id=student_id, session_id=session_id)

    # STEP: START_CHAPTER -> send first chunk
    if step == "START_CHAPTER":
        mem = get_memories(student_id)
        name = mem.get("student_name", "friend")
        idx = int(mem.get("chunk_index", "0") or "0")
        set_step(student_id, "TEACHING")
        msg = teacher_teach_chunk(mem, name, idx)
        save_message(session_id, student_id, "teacher", msg, meta)
        return RespondResponse(text=msg, student_id=student_id, session_id=session_id)

    # STEP: TEACHING -> evaluate answer -> next chunk or remediate
    if step == "TEACHING":
        mem = get_memories(student_id)
        name = mem.get("student_name", "friend")
        idx = int(mem.get("chunk_index", "0") or "0")
        chunk = DEMO_CHUNKS[min(idx, len(DEMO_CHUNKS) - 1)]

        # repeated-question handling
        last_q = mem.get("last_check_q", "")
        if last_q and student_text.strip().lower() == last_q.strip().lower():
            # student repeated same text; respond with alternative explanation
            msg = teacher_remediate(mem, name, idx)
            save_message(session_id, student_id, "teacher", msg, meta)
            return RespondResponse(text=msg, student_id=student_id, session_id=session_id)

        # Store last student answer to detect repeats
        set_memory(student_id, "last_student_answer", student_text.strip(), 0.7)

        if is_answer_correct(student_text, chunk["expected_keywords"]):
            # move next chunk
            next_idx = idx + 1
            set_memory(student_id, "chunk_index", str(next_idx), 0.95)

            if next_idx >= len(DEMO_CHUNKS):
                # end demo teaching
                end_msg = say(
                    f"Excellent, {name}! You finished today’s demo lesson. Want a quick revision or a fun quiz?",
                    f"Excellent, {name}! Aaj ka demo lesson complete. Quick revision karna hai ya fun quiz?",
                    mem.get("preferred_language_mode", "EN"),
                )
                set_step(student_id, "QNA")
                save_message(session_id, student_id, "teacher", end_msg, meta)
                return RespondResponse(text=end_msg, student_id=student_id, session_id=session_id)

            # praise + next chunk
            praise = say(
                f"Great job, {name}! ✅",
                f"Bahut badiya, {name}! ✅",
                mem.get("preferred_language_mode", "EN"),
            )
            msg = praise + "\n\n" + teacher_teach_chunk(mem, name, next_idx)
            save_message(session_id, student_id, "teacher", msg, meta)
            return RespondResponse(text=msg, student_id=student_id, session_id=session_id)

        # wrong/unclear -> remediate same chunk
        set_memory(student_id, "last_check_q", student_text.strip(), 0.6)
        msg = teacher_remediate(mem, name, idx)
        save_message(session_id, student_id, "teacher", msg, meta)
        return RespondResponse(text=msg, student_id=student_id, session_id=session_id)

    # STEP: QNA or fallback -> use LLM if available, otherwise a helpful reply
    mem = get_memories(student_id)
    name = mem.get("student_name", "friend")
    lang_mode = mem.get("preferred_language_mode", "EN")

    # If student asks something during QNA, respond and keep it interactive
    if get_step(mem) == "QNA":
        # Lightweight: answer briefly + ask a small followup
        base = say(
            f"Okay {name}. {student_text}",
            f"Okay {name}. {student_text}",
            lang_mode,
        )
        msg = base + "\n\n" + say(
            "Do you want a quick revision or one quiz question?",
            "Quick revision karni hai ya ek quiz question?",
            lang_mode,
        )
        save_message(session_id, student_id, "teacher", msg, meta)
        return RespondResponse(text=msg, student_id=student_id, session_id=session_id)

    # Last resort: LLM (optional)
    recent = get_recent_messages(student_id, session_id, limit=12)
    chat_msgs: List[Dict[str, str]] = []
    for m in recent:
        if m["role"] == "student":
            chat_msgs.append({"role": "user", "content": m["content"]})
        else:
            chat_msgs.append({"role": "assistant", "content": m["content"]})

    system = build_demo_system_prompt(meta, mem)
    try:
        teacher_text = await llm_reply(system, chat_msgs)
    except Exception:
        teacher_text = say(
            f"Okay {name}. Can you repeat in one simple line?",
            f"Okay {name}. Ek simple line mein phir se batao.",
            lang_mode,
        )

    save_message(session_id, student_id, "teacher", teacher_text, meta)
    return RespondResponse(text=teacher_text, student_id=student_id, session_id=session_id)


@app.get("/history")
def history(student_id: str, session_id: str, limit: int = 50):
    limit = max(1, min(int(limit), 200))
    msgs = get_recent_messages(student_id, session_id, limit=limit)
    return {"student_id": student_id, "session_id": session_id, "messages": msgs}