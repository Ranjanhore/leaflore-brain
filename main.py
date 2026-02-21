import os, json, time
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# Optional Redis (recommended). If not set, fallback to in-memory dict.
REDIS_URL = os.getenv("REDIS_URL")
memory_fallback: Dict[str, Any] = {}

try:
    import redis
    rdb = redis.from_url(REDIS_URL) if REDIS_URL else None
except Exception:
    rdb = None

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "450"))


# ---------- Models ----------
class VoiceSettings(BaseModel):
    mode: Optional[str] = "full_duplex"
    barge_in: Optional[bool] = True

class BrainRequest(BaseModel):
    session_id: str
    student_id: str
    board: str
    grade: str
    subject: str
    chapter: str
    concept: str
    chunk_id: Optional[str] = None
    class_phase: Optional[str] = "teach"  # teach | quiz | recap
    student_input: str
    signals: Optional[Dict[str, Any]] = None
    voice: Optional[VoiceSettings] = None

class BrainResponse(BaseModel):
    teacher_text: str
    tts_text: str
    next_action: str
    question: Optional[str] = None
    quiz: Optional[Dict[str, Any]] = None
    difficulty: str
    memory_update: Optional[Dict[str, Any]] = None
    chunk: Dict[str, Any]


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Memory helpers ----------
def mem_get(session_id: str) -> Dict[str, Any]:
    key = f"session:{session_id}"
    if rdb:
        raw = rdb.get(key)
        if raw:
            return json.loads(raw)
        return {"facts": [], "preferences": {}, "mastery": {}, "last_question": None, "last_chunk_id": None}
    return memory_fallback.get(key, {"facts": [], "preferences": {}, "mastery": {}, "last_question": None, "last_chunk_id": None})

def mem_set(session_id: str, obj: Dict[str, Any]):
    key = f"session:{session_id}"
    if rdb:
        rdb.setex(key, 60 * 60 * 24 * 14, json.dumps(obj))  # 14 days
    else:
        memory_fallback[key] = obj


# ---------- Logic helpers ----------
def decide_difficulty(req: BrainRequest, mem: Dict[str, Any]) -> str:
    emotion = (req.signals or {}).get("emotion", "neutral")
    engagement = (req.signals or {}).get("engagement", "normal")
    mastery = (mem.get("mastery") or {}).get(req.concept.lower(), 0.0)

    if emotion in ["confused", "anxious"] or mastery < 0.35:
        return "easy"
    if engagement in ["high", "curious"] and mastery > 0.65:
        return "hard"
    return "normal"

def should_quiz(req: BrainRequest, mem: Dict[str, Any]) -> bool:
    mastery = (mem.get("mastery") or {}).get(req.concept.lower(), 0.0)
    # quiz if mastery moderate or student says understood
    if "understood" in req.student_input.lower() or "got it" in req.student_input.lower():
        return True
    return mastery >= 0.55 and req.class_phase != "quiz"

def chunk_progress(req: BrainRequest, mem: Dict[str, Any]) -> Dict[str, Any]:
    # super simple default (you can replace with your DB order)
    current = req.chunk_id or mem.get("last_chunk_id") or "chunk_01"
    # naive next chunk naming
    next_chunk = None
    try:
        if current.startswith("chunk_"):
            n = int(current.split("_")[1])
            next_chunk = f"chunk_{n+1:02d}"
    except:
        next_chunk = None

    # advance if student is not confused and they said ok/yes
    emotion = (req.signals or {}).get("emotion", "neutral")
    should_advance = ("yes" in req.student_input.lower() or "ok" in req.student_input.lower()) and emotion != "confused"

    return {"current_chunk_id": current, "next_chunk_id": next_chunk, "should_advance": should_advance}

def tts_clean(text: str) -> str:
    # very light cleanup
    t = text.replace("**", "").replace("*", "").replace("•", "")
    return t.strip()


# ---------- Main endpoint ----------
@app.post("/respond", response_model=BrainResponse)
def respond(req: BrainRequest):
    if not req.student_input:
        raise HTTPException(status_code=400, detail="student_input is required")

    mem = mem_get(req.session_id)
    difficulty = decide_difficulty(req, mem)
    do_quiz = should_quiz(req, mem)
    chunk_info = chunk_progress(req, mem)

    pace = (req.signals or {}).get("pace_preference") or mem.get("preferences", {}).get("pace", "slow")
    barge_in = (req.voice.barge_in if req.voice else True)

    # Voice-first system prompt
    system_prompt = f"""
You are Leaflore AI Teacher.

Context:
Board: {req.board}
Grade: {req.grade}
Subject: {req.subject}
Chapter: {req.chapter}
Concept: {req.concept}
Difficulty: {difficulty}
Pace: {pace}

Memory (what you already know):
Facts: {mem.get("facts", [])}
Preferences: {mem.get("preferences", {})}
Last question: {mem.get("last_question")}

Voice rules:
- Speak in short sentences.
- Keep response 10 to 25 seconds.
- End with ONE short question.
- Add a clear yield point ("Your turn...").
- If barge-in is enabled ({barge_in}), avoid long monologues.

Output format MUST be JSON with keys:
teacher_text, tts_text, next_action, question, quiz, memory_update, chunk

Where:
- next_action is one of: "listen", "speak", "wait_for_student"
- quiz can be null or an object with type/prompt/options/answer_index
- memory_update contains facts[] and preferences{} updates (only if needed)
"""

    user_payload = {
        "class_phase": req.class_phase,
        "chunk_id": req.chunk_id,
        "student_input": req.student_input,
        "signals": req.signals or {},
        "do_quiz": do_quiz,
        "chunk": chunk_info,
    }

    start = time.time()
    r = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
        max_output_tokens=MAX_TOKENS,
    )

    raw = r.output_text if hasattr(r, "output_text") else ""
    if not raw:
        raise HTTPException(status_code=500, detail="empty model response")

    # Parse JSON safely
    try:
        data = json.loads(raw)
    except Exception:
        # fallback: wrap plain text into structure
        data = {
            "teacher_text": raw,
            "tts_text": tts_clean(raw),
            "next_action": "wait_for_student",
            "question": None,
            "quiz": None,
            "memory_update": None,
            "chunk": chunk_info,
        }

    # Ensure required fields
    teacher_text = data.get("teacher_text", "")
    tts_text = data.get("tts_text", tts_clean(teacher_text))
    next_action = data.get("next_action", "wait_for_student")
    question = data.get("question")
    quiz = data.get("quiz")
    memory_update = data.get("memory_update") or {}
    chunk = data.get("chunk") or chunk_info

    # Apply memory updates
    if memory_update:
        facts = memory_update.get("facts", [])
        prefs = memory_update.get("preferences", {})
        if facts:
            mem["facts"] = list(dict.fromkeys(mem.get("facts", []) + facts))[-50:]
        if prefs:
            mem["preferences"] = {**mem.get("preferences", {}), **prefs}
        if question:
            mem["last_question"] = question
        mem["last_chunk_id"] = chunk.get("current_chunk_id", mem.get("last_chunk_id"))
        # tiny mastery bump if student says ok/understood
        if "understood" in req.student_input.lower() or "got it" in req.student_input.lower():
            ckey = req.concept.lower()
            mem["mastery"][ckey] = min(1.0, float(mem.get("mastery", {}).get(ckey, 0.0)) + 0.15)

        mem_set(req.session_id, mem)

    # (Optional) Log latency/cost estimate later
    _lat = time.time() - start

    return BrainResponse(
        teacher_text=teacher_text,
        tts_text=tts_text,
        next_action=next_action,
        question=question,
        quiz=quiz,
        difficulty=difficulty,
        memory_update=memory_update or None,
        chunk=chunk,
    )
