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


def set_memory(student_id: str, key: str, value: str):
    conn = _db()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO memories(student_id,key,value)
        VALUES(?,?,?)
        ON CONFLICT(student_id,key)
        DO UPDATE SET value=excluded.value
    """, (student_id, key, value))
    conn.commit()
    conn.close()


def get_memories(student_id: str) -> Dict[str, str]:
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT key,value FROM memories WHERE student_id=?", (student_id,))
    rows = cur.fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}


def save_message(session_id: str, student_id: str, role: str, content: str):
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
    action: str = "respond"
    student_input: str = ""
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
def greeting():
    h = time.localtime().tm_hour
    if 5 <= h < 12:
        return "Good morning"
    if 12 <= h < 17:
        return "Good afternoon"
    if 17 <= h < 22:
        return "Good evening"
    return "Hello"


def pick_lang(text: str):
    t = text.lower()
    if "hindi" in t:
        return "HI"
    if "both" in t:
        return "BOTH"
    if "english" in t:
        return "EN"
    return ""


def say(en: str, hi: str, mode: str):
    if mode == "HI":
        return hi
    if mode == "BOTH":
        return en + " " + hi
    return en


def correct(ans: str, keys: List[str]):
    a = ans.lower()
    return any(k.lower() in a for k in keys)


# ----------------------------
# DEMO CHUNKS (REPLACED FULLY)
# ----------------------------
DEMO_CHUNKS = [

    {
        "title": "Leaf Structure",
        "teach_en": "A leaf is the food factory of a plant. The flat green part is called lamina. The main line in the middle is midrib. Small lines are veins. The stalk is petiole.",
        "teach_hi": "Leaf plant ka food factory hota hai. Flat green part ko lamina kehte hain. Beech ki line midrib hoti hai. Chhoti lines veins hoti hain. Dandi ko petiole kehte hain.",
        "question_en": "What is the middle line of a leaf called?",
        "question_hi": "Leaf ki beech wali line ko kya kehte hain?",
        "keywords": ["midrib"]
    },

    {
        "title": "Types of Leaves",
        "teach_en": "Some leaves are simple like mango. Some are compound like neem. Compound leaves have many small leaflets.",
        "teach_hi": "Kuch leaves simple hoti hain jaise aam. Kuch compound hoti hain jaise neem. Compound leaves mein chhote leaflets hote hain.",
        "question_en": "Neem leaf is simple or compound?",
        "question_hi": "Neem simple hai ya compound?",
        "keywords": ["compound"]
    },

    {
        "title": "Venation",
        "teach_en": "Venation means pattern of veins. Net pattern is reticulate. Side by side pattern is parallel.",
        "teach_hi": "Venation matlab veins ka pattern. Net pattern reticulate hota hai. Side by side parallel hota hai.",
        "question_en": "Grass has reticulate or parallel venation?",
        "question_hi": "Grass mein reticulate hota hai ya parallel?",
        "keywords": ["parallel"]
    },

    {
        "title": "Leaf Modification",
        "teach_en": "Cactus leaves become spines to save water. Some leaves become tendrils to climb.",
        "teach_hi": "Cactus ki leaves pani bachane ke liye spines ban jaati hain. Kuch leaves climbing ke liye tendrils ban jaati hain.",
        "question_en": "Why do cactus leaves become spines?",
        "question_hi": "Cactus ki leaves spines kyun ban jaati hain?",
        "keywords": ["water"]
    }
]

# ----------------------------
# ROUTE
# ----------------------------
@app.post("/respond", response_model=RespondResponse)
async def respond(req: RespondRequest, request: Request):

    student_id = req.student_id or "anonymous"
    session_id = req.session_id or "default"

    mem = get_memories(student_id)

    # START CLASS
    if req.action == "start_class":

        set_memory(student_id, "step", "ASK_NAME")
        set_memory(student_id, "chunk_index", "0")

        subject = req.subject or "Science"

        text = f"{greeting()}! Welcome to Leaflore. My name is Anaya. I am your {subject} teacher. What is your name? To speak with me, click the Speak button below this screen."

        save_message(session_id, student_id, "teacher", text)
        return RespondResponse(text=text, student_id=student_id, session_id=session_id)

    step = mem.get("step", "ASK_NAME")

    # ASK NAME
    if step == "ASK_NAME":
        name = req.student_input.strip().title()
        set_memory(student_id, "student_name", name)
        set_memory(student_id, "step", "ASK_LANG")
        text = f"Nice to meet you, {name}! Which language are you comfortable in: English, Hindi or Both?"
        save_message(session_id, student_id, "teacher", text)
        return RespondResponse(text=text, student_id=student_id, session_id=session_id)

    # ASK LANGUAGE
    if step == "ASK_LANG":
        mode = pick_lang(req.student_input)
        if not mode:
            return RespondResponse(text="Please choose: English, Hindi or Both.", student_id=student_id, session_id=session_id)

        set_memory(student_id, "lang", mode)
        set_memory(student_id, "step", "TEACH")

        name = mem.get("student_name", "friend")
        chapter = req.chapter or "The Leaf"

        intro = say(
            f"Great {name}! Today we will learn {chapter}. This is a one hour class. To ask questions click Speak button. To stop click Stop button on top right. Time starts now.",
            f"Bahut badiya {name}! Aaj hum {chapter} seekhenge. Ye one hour class hai. Question poochne ke liye Speak dabao. Stop karne ke liye upar Stop dabao. Time start.",
            mode
        )

        save_message(session_id, student_id, "teacher", intro)
        return RespondResponse(text=intro, student_id=student_id, session_id=session_id)

    # TEACHING
    if step == "TEACH":

        mode = mem.get("lang", "EN")
        name = mem.get("student_name", "friend")
        index = int(mem.get("chunk_index", "0"))

        if index >= len(DEMO_CHUNKS):
            text = say(
                f"Excellent {name}! Demo class completed. Do you want revision or quiz?",
                f"Excellent {name}! Demo class complete. Revision karna hai ya quiz?",
                mode
            )
            save_message(session_id, student_id, "teacher", text)
            return RespondResponse(text=text, student_id=student_id, session_id=session_id)

        chunk = DEMO_CHUNKS[index]

        # If student answered previous question
        if req.student_input:
            prev_index = index - 1
            if prev_index >= 0:
                prev_chunk = DEMO_CHUNKS[prev_index]
                if not correct(req.student_input, prev_chunk["keywords"]):
                    retry = say(
                        f"No worries {name}, let me explain again differently.",
                        f"Koi baat nahi {name}, main aur simple samjhati hoon.",
                        mode
                    )
                    save_message(session_id, student_id, "teacher", retry)
                    return RespondResponse(text=retry, student_id=student_id, session_id=session_id)

        teaching = say(chunk["teach_en"], chunk["teach_hi"], mode)
        question = say(chunk["question_en"], chunk["question_hi"], mode)

        set_memory(student_id, "chunk_index", str(index + 1))

        full = f"{teaching}\n\n{question}"
        save_message(session_id, student_id, "teacher", full)
        return RespondResponse(text=full, student_id=student_id, session_id=session_id)

    return RespondResponse(text="Let’s continue learning.", student_id=student_id, session_id=session_id)