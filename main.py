# main.py
from __future__ import annotations

import os
import re
import json
import time
import sqlite3
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Request
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
# DB (SQLite for demo memory/state)
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
    CREATE TABLE IF NOT EXISTS memories (
      student_id TEXT,
      key TEXT,
      value TEXT,
      PRIMARY KEY(student_id, key)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      student_id TEXT,
      role TEXT,
      content TEXT,
      created_at INTEGER
    );
    """)

    conn.commit()
    conn.close()


init_db()


def set_memory(student_id: str, key: str, value: str) -> None:
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO memories(student_id,key,value)
        VALUES(?,?,?)
        ON CONFLICT(student_id,key)
        DO UPDATE SET value=excluded.value
        """,
        (student_id, key, value),
    )
    conn.commit()
    conn.close()


def get_memories(student_id: str) -> Dict[str, str]:
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT key,value FROM memories WHERE student_id=?", (student_id,))
    rows = cur.fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}


def save_message(session_id: str, student_id: str, role: str, content: str) -> None:
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages(session_id,student_id,role,content,created_at) VALUES(?,?,?,?,?)",
        (session_id, student_id, role, content, int(time.time())),
    )
    conn.commit()
    conn.close()


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


class RespondResponse(BaseModel):
    text: str
    student_id: str
    session_id: str


# ----------------------------
# Helpers
# ----------------------------
NAME_RE = re.compile(r"\b(my name is|i am|i'm)\s+([A-Za-z][A-Za-z\s]{1,30})\b", re.I)


def greeting() -> str:
    h = time.localtime().tm_hour
    if 5 <= h < 12:
        return "Good morning"
    if 12 <= h < 17:
        return "Good afternoon"
    if 17 <= h < 22:
        return "Good evening"
    return "Hello"


def normalize(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\bhell teacher\b", "hello teacher", t, flags=re.I)
    t = re.sub(r"\bhelo\b", "hello", t, flags=re.I)
    t = re.sub(r"\bteachear\b", "teacher", t, flags=re.I)
    return t


def pick_lang(text: str) -> str:
    t = (text or "").strip().lower()
    if "both" in t or "hinglish" in t or ("english" in t and "hindi" in t):
        return "BOTH"
    if "hindi" in t or t in {"hi", "h"}:
        return "HI"
    if "english" in t or t in {"en", "e"}:
        return "EN"
    return ""


def say(en: str, hi: str, mode: str) -> str:
    # BOTH = Hinglish full time (English + Hindi)
    if mode == "HI":
        return hi or en
    if mode == "BOTH":
        if hi:
            return f"{en} {hi}".strip()
        return en
    return en


def try_extract_name(text: str) -> str:
    t = (text or "").strip()
    m = NAME_RE.search(t)
    if m:
        return m.group(2).strip().title()
    words = re.findall(r"[A-Za-z]{2,20}", t)
    if not words:
        return ""
    bad = {"english", "hindi", "both", "science", "teacher"}
    if words[0].lower() in bad:
        return ""
    return words[0].title()


def correct(ans: str, keys: List[str]) -> bool:
    a = (ans or "").lower()
    return any((k or "").lower() in a for k in keys if k)


def safe_int(s: str, default: int = 0) -> int:
    try:
        return int(s)
    except Exception:
        return default


# ----------------------------
# Supabase Chunk Loader
# ----------------------------
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
SUPABASE_SERVICE_ROLE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()

# Table expected: chapter_chunks
# Minimal columns supported by this loader:
# - board (text), grade (text), subject (text), chapter (text)
# - seq (int)
# - title (text)
# - teach_en (text) OR chunk_text (text)
# - teach_hi (text) optional
# - question_en (text) optional
# - question_hi (text) optional
# - expected_keywords (text) optional  (csv OR json array)
# - is_active (bool)
#
# It will fall back safely if some columns are missing.


def _parse_keywords(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        # try JSON list
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if str(x).strip()]
            except Exception:
                pass
        # csv
        return [p.strip() for p in s.split(",") if p.strip()]
    return []


async def load_chunks_from_supabase(board: str, grade: str, subject: str, chapter: str) -> List[Dict[str, Any]]:
    """
    Loads ordered chunks from Supabase REST.
    Uses service role key (backend only). Never expose this key to frontend.
    Returns a list in the same shape used by the teaching loop.
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        return []

    # Lazy import: keep dependencies minimal (httpx is already used in your project earlier)
    import httpx

    base = SUPABASE_URL.rstrip("/")
    url = f"{base}/rest/v1/chapter_chunks"

    # We request a superset of fields; Supabase will return only existing columns.
    # If some columns don't exist, it still works as long as the table exists.
    select = ",".join(
        [
            "board",
            "grade",
            "subject",
            "chapter",
            "seq",
            "title",
            "teach_en",
            "teach_hi",
            "chunk_text",
            "question_en",
            "question_hi",
            "check_q_en",
            "check_q_hi",
            "expected_keywords",
            "is_active",
        ]
    )

    params = {
        "select": select,
        "board": f"eq.{board}",
        "grade": f"eq.{grade}",
        "subject": f"eq.{subject}",
        "chapter": f"eq.{chapter}",
        "is_active": "eq.true",
        "order": "seq.asc",
    }

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, params=params, headers=headers)
            if r.status_code >= 400:
                # minimal log only
                print(f"[Supabase] chunk load failed: {r.status_code} {r.text[:200]}")
                return []
            rows = r.json()
            if not isinstance(rows, list):
                return []
    except Exception as e:
        print(f"[Supabase] chunk load exception: {e}")
        return []

    chunks: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        title = (row.get("title") or "").strip() or "Chunk"
        teach_en = (row.get("teach_en") or row.get("chunk_text") or "").strip()
        teach_hi = (row.get("teach_hi") or "").strip()

        # support either question_en/question_hi OR check_q_en/check_q_hi
        q_en = (row.get("question_en") or row.get("check_q_en") or "").strip()
        q_hi = (row.get("question_hi") or row.get("check_q_hi") or "").strip()

        seq = safe_int(str(row.get("seq") or "0"), 0)

        keywords = _parse_keywords(row.get("expected_keywords"))
        # if keywords missing, try to auto-create from question (very weak fallback)
        if not keywords and q_en:
            keywords = []

        chunks.append(
            {
                "seq": seq,
                "title": title,
                "teach_en": teach_en,
                "teach_hi": teach_hi,
                "question_en": q_en,
                "question_hi": q_hi,
                "keywords": keywords,
            }
        )

    # Ensure stable order even if seq missing
    chunks.sort(key=lambda c: (c.get("seq", 0), c.get("title", "")))
    return chunks


def get_cached_chunks(mem: Dict[str, str]) -> List[Dict[str, Any]]:
    raw = mem.get("chunks_json", "")
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except Exception:
        return []
    return []


def cache_chunks(student_id: str, chunks_key: str, chunks: List[Dict[str, Any]]) -> None:
    set_memory(student_id, "chunks_key", chunks_key)
    set_memory(student_id, "chunks_json", json.dumps(chunks, ensure_ascii=False))


# ----------------------------
# DEMO fallback chunks (used only if Supabase has no chunks yet)
# ----------------------------
DEMO_CHUNKS: List[Dict[str, Any]] = [
    {
        "seq": 1,
        "title": "Leaf Structure",
        "teach_en": "A leaf is the food factory of a plant. The flat green part is called lamina. The main line in the middle is midrib. Small lines are veins. The stalk is petiole.",
        "teach_hi": "Leaf plant ka food factory hota hai. Flat green part ko lamina kehte hain. Beech ki line midrib hoti hai. Chhoti lines veins hoti hain. Dandi ko petiole kehte hain.",
        "question_en": "What is the middle line of a leaf called?",
        "question_hi": "Leaf ki beech wali line ko kya kehte hain?",
        "keywords": ["midrib"],
    },
    {
        "seq": 2,
        "title": "Types of Leaves",
        "teach_en": "Some leaves are simple like mango. Some are compound like neem. Compound leaves have many small leaflets.",
        "teach_hi": "Kuch leaves simple hoti hain jaise aam. Kuch compound hoti hain jaise neem. Compound leaves mein chhote leaflets hote hain.",
        "question_en": "Neem leaf is simple or compound?",
        "question_hi": "Neem simple hai ya compound?",
        "keywords": ["compound"],
    },
    {
        "seq": 3,
        "title": "Venation",
        "teach_en": "Venation means pattern of veins. Net pattern is reticulate. Side by side pattern is parallel.",
        "teach_hi": "Venation matlab veins ka pattern. Net pattern reticulate hota hai. Side by side parallel hota hai.",
        "question_en": "Grass has reticulate or parallel venation?",
        "question_hi": "Grass mein reticulate hota hai ya parallel?",
        "keywords": ["parallel"],
    },
    {
        "seq": 4,
        "title": "Leaf Modification",
        "teach_en": "Cactus leaves become spines to save water. Some leaves become tendrils to climb.",
        "teach_hi": "Cactus ki leaves pani bachane ke liye spines ban jaati hain. Kuch leaves climbing ke liye tendrils ban jaati hain.",
        "question_en": "Why do cactus leaves become spines?",
        "question_hi": "Cactus ki leaves spines kyun ban jaati hain?",
        "keywords": ["water", "save", "loss"],
    },
]


def _get_selected(req: RespondRequest, mem: Dict[str, str]) -> Dict[str, str]:
    board = (req.board or mem.get("selected_board") or "ICSE").strip()
    grade = (req.grade or mem.get("selected_grade") or "6").strip()
    subject = (req.subject or mem.get("selected_subject") or "Science").strip()
    chapter = (req.chapter or mem.get("selected_chapter") or "The Leaf").strip()
    return {"board": board, "grade": grade, "subject": subject, "chapter": chapter}


def _teach_chunk_text(mode: str, student_name: str, chunk: Dict[str, Any]) -> str:
    # tiny storytelling hook (kept short)
    hook_en = f"Okay {student_name}, imagine you are a tiny explorer inside this chapter."
    hook_hi = f"Okay {student_name}, socho tum ek tiny explorer ho is chapter ke andar."
    hook = say(hook_en, hook_hi, mode)

    teach = say(chunk.get("teach_en", "") or "", chunk.get("teach_hi", "") or "", mode).strip()
    q = say(chunk.get("question_en", "") or "", chunk.get("question_hi", "") or "", mode).strip()

    out = hook
    if teach:
        out += " " + teach
    if q:
        out += "\n\n" + q
    return out.strip()


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/respond", response_model=RespondResponse)
async def respond(req: RespondRequest, request: Request):
    student_id = (req.student_id or "").strip() or "anonymous"
    session_id = (req.session_id or "").strip() or "default"

    mem = get_memories(student_id)
    sel = _get_selected(req, mem)

    # persist selections
    set_memory(student_id, "selected_board", sel["board"])
    set_memory(student_id, "selected_grade", sel["grade"])
    set_memory(student_id, "selected_subject", sel["subject"])
    set_memory(student_id, "selected_chapter", sel["chapter"])

    student_text = normalize(req.student_input)

    # ----------------------------
    # START CLASS: load chunks from Supabase and cache them
    # ----------------------------
    if (req.action or "").strip() == "start_class":
        set_memory(student_id, "step", "ASK_NAME")
        set_memory(student_id, "chunk_index", "0")

        chunks_key = f'{sel["board"]}|{sel["grade"]}|{sel["subject"]}|{sel["chapter"]}'
        chunks: List[Dict[str, Any]] = []

        # Load from Supabase first
        chunks = await load_chunks_from_supabase(sel["board"], sel["grade"], sel["subject"], sel["chapter"])
        if not chunks:
            # fallback to demo chunks (so class never breaks)
            chunks = DEMO_CHUNKS

        cache_chunks(student_id, chunks_key, chunks)

        text = (
            f"{greeting()}! Welcome to Leaflore. My name is Anaya. "
            f'I am your {sel["subject"]} teacher. What is your name? '
            "To speak with me, click the Speak button below this screen."
        )
        save_message(session_id, student_id, "teacher", text)
        return RespondResponse(text=text, student_id=student_id, session_id=session_id)

    # Save student message (non-empty)
    if student_text:
        save_message(session_id, student_id, "student", student_text)

    step = mem.get("step", "ASK_NAME")

    # ASK NAME
    if step == "ASK_NAME":
        name = try_extract_name(student_text) or student_text.strip().title()
        name = (name or "Friend").strip()
        set_memory(student_id, "student_name", name)
        set_memory(student_id, "step", "ASK_LANG")
        text = f"Nice to meet you, {name}! Which language are you comfortable in: English, Hindi or Both?"
        save_message(session_id, student_id, "teacher", text)
        return RespondResponse(text=text, student_id=student_id, session_id=session_id)

    # ASK LANGUAGE
    if step == "ASK_LANG":
        mode = pick_lang(student_text)
        if not mode:
            text = "Please choose: English, Hindi or Both."
            save_message(session_id, student_id, "teacher", text)
            return RespondResponse(text=text, student_id=student_id, session_id=session_id)

        set_memory(student_id, "lang", mode)
        set_memory(student_id, "step", "TEACH")

        mem = get_memories(student_id)
        name = mem.get("student_name", "friend")

        intro = say(
            f"Great {name}! Today we will learn “{sel['chapter']}” in {sel['subject']}. "
            "This is a one hour class. To ask questions click the Speak button below. "
            "To stop, click Stop at top right. Time starts now. Let’s begin!",
            f"Bahut badiya {name}! Aaj hum “{sel['chapter']}” {sel['subject']} mein seekhenge. "
            "Ye one hour class hai. Question poochne ke liye neeche Speak button dabao. "
            "Stop karna ho to top right pe Stop dabao. Time start. Chaliye shuru karte hain!",
            mode,
        )
        save_message(session_id, student_id, "teacher", intro)
        return RespondResponse(text=intro, student_id=student_id, session_id=session_id)

    # TEACHING
    if step == "TEACH":
        mem = get_memories(student_id)
        mode = mem.get("lang", "EN")
        name = mem.get("student_name", "friend")

        # ensure cached chunks match current selection; reload if changed
        chunks_key = f'{sel["board"]}|{sel["grade"]}|{sel["subject"]}|{sel["chapter"]}'
        cached_key = mem.get("chunks_key", "")
        chunks = get_cached_chunks(mem)

        if not chunks or cached_key != chunks_key:
            chunks = await load_chunks_from_supabase(sel["board"], sel["grade"], sel["subject"], sel["chapter"])
            if not chunks:
                chunks = DEMO_CHUNKS
            cache_chunks(student_id, chunks_key, chunks)

        idx = safe_int(mem.get("chunk_index", "0"), 0)

        # If student just answered, validate against previous chunk question
        if student_text and idx > 0:
            prev = chunks[min(idx - 1, len(chunks) - 1)]
            expected = prev.get("keywords") or prev.get("expected_keywords") or []
            if isinstance(expected, str):
                expected = _parse_keywords(expected)
            if not isinstance(expected, list):
                expected = []
            if expected and not correct(student_text, expected):
                retry = say(
                    f"No worries {name}, let me explain that in a simpler way with a quick example. Then try again.",
                    f"Koi baat nahi {name}, main simple example ke saath phir se samjhati hoon. Phir aap try karo.",
                    mode,
                )
                # Repeat the same question again (do not advance idx)
                q = say(
                    prev.get("question_en", "") or "Try again:",
                    prev.get("question_hi", "") or "Phir se try karo:",
                    mode,
                )
                text = (retry + "\n\n" + q).strip()
                save_message(session_id, student_id, "teacher", text)
                return RespondResponse(text=text, student_id=student_id, session_id=session_id)

        # End condition
        if idx >= len(chunks):
            end_text = say(
                f"Excellent {name}! Demo class completed. Do you want a quick revision or a fun quiz?",
                f"Excellent {name}! Demo class complete. Quick revision karni hai ya fun quiz?",
                mode,
            )
            save_message(session_id, student_id, "teacher", end_text)
            return RespondResponse(text=end_text, student_id=student_id, session_id=session_id)

        # Teach next chunk
        chunk = chunks[idx]
        text = _teach_chunk_text(mode, name, chunk)

        # Advance pointer
        set_memory(student_id, "chunk_index", str(idx + 1))

        save_message(session_id, student_id, "teacher", text)
        return RespondResponse(text=text, student_id=student_id, session_id=session_id)

    # Fallback (should rarely happen)
    text = "Let’s continue learning."
    save_message(session_id, student_id, "teacher", text)
    return RespondResponse(text=text, student_id=student_id, session_id=session_id)
