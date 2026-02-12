import os
import json
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client, Client


# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(title="Leaflore Brain API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CLIENTS
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var")

client = OpenAI(api_key=OPENAI_API_KEY)

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Optional[Client] = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None


# ============================================================
# REQUEST MODEL
# ============================================================

class BrainRequest(BaseModel):
    student_id: Optional[str] = "demo"
    chapter: Optional[str] = ""
    concept: Optional[str] = ""
    student_input: str
    signals: Optional[Dict[str, Any]] = {}
    brain: Optional[Dict[str, Any]] = {}


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Leaflore NeuroMentor — an elite, warm, calm, highly intelligent science teacher.

You are NOT a medical doctor.
No diagnosis. No therapy language. No psychiatric labels.

Return ONLY valid JSON with EXACT keys:

text
mode
next_action
micro_q
ui_hint
memory_update

No markdown.
No extra keys.

Always end with exactly ONE micro-question.

Confidence and stress must change max ±10 per turn.
"""


# ============================================================
# SUPABASE HELPERS
# ============================================================

def _require_supabase():
    if supabase is None:
        raise HTTPException(
            status_code=500,
            detail="Supabase is not configured (missing env vars)."
        )

def load_brain(student_id: str) -> Dict[str, Any]:
    _require_supabase()
    res = (
        supabase.table("student_brain")
        .select("brain_data")
        .eq("student_id", student_id)
        .limit(1)
        .execute()
    )

    if res.data and len(res.data) > 0:
        return res.data[0].get("brain_data") or {}

    return {}

def save_brain(student_id: str, brain_data: Dict[str, Any]) -> None:
    _require_supabase()
    if not isinstance(brain_data, dict):
        brain_data = {}

    supabase.table("student_brain").upsert(
        {
            "student_id": student_id,
            "brain_data": brain_data
        }
    ).execute()

def merge_brain(existing: Dict[str, Any], memory_update: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(existing, dict):
        existing = {}
    if not isinstance(memory_update, dict):
        memory_update = {}

    merged = dict(existing)
    merged.update(memory_update)
    return merged


# ============================================================
# CONFIDENCE AUTO-CLAMPING
# ============================================================

def _clamp(n: int, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, int(n)))

def _limit_delta(new_val: int, old_val: int, max_delta: int = 10) -> int:
    if new_val > old_val + max_delta:
        return old_val + max_delta
    if new_val < old_val - max_delta:
        return old_val - max_delta
    return new_val

def apply_score_clamp(existing: Dict[str, Any], memory_update: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(existing, dict):
        existing = {}
    if not isinstance(memory_update, dict):
        memory_update = {}

    out = dict(memory_update)

    for key in ("confidence_score", "stress_score"):
        if key in memory_update:
            try:
                new_raw = int(memory_update[key])
            except Exception:
                continue

            old_raw = existing.get(key, 50)

            try:
                old_raw = int(old_raw)
            except Exception:
                old_raw = 50

            new_raw = _clamp(new_raw)
            old_raw = _clamp(old_raw)

            out[key] = _limit_delta(new_raw, old_raw)

    return out


# ============================================================
# ROOT & HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "leaflore-brain",
        "message": "API is running. Use /docs or POST /respond"
    }

@app.get("/health")
def health():
    return {"ok": True}


# ============================================================
# BRAIN ENDPOINTS
# ============================================================

@app.get("/student/{student_id}/brain")
def get_student_brain(student_id: str):
    brain = load_brain(student_id)
    return {"student_id": student_id, "brain": brain}

@app.post("/student/{student_id}/brain")
def set_student_brain(student_id: str, brain: Dict[str, Any]):
    save_brain(student_id, brain)
    return {"ok": True, "student_id": student_id}


# ============================================================
# MAIN AI ENDPOINT
# ============================================================

@app.post("/respond")
def respond(req: BrainRequest):
    try:
        student_id = (req.student_id or "demo").strip()

        brain = req.brain or {}
        if not brain and supabase:
            brain = load_brain(student_id)

        user_context = f"""
Chapter: {req.chapter}
Concept: {req.concept}
Student Input: {req.student_input}
Signals: {json.dumps(req.signals, ensure_ascii=False)}
Brain Memory: {json.dumps(brain, ensure_ascii=False)}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.6,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_context}
            ]
        )

        content = (response.choices[0].message.content or "").strip()

        parsed = json.loads(content)

        required_keys = {
            "text",
            "mode",
            "next_action",
            "micro_q",
            "ui_hint",
            "memory_update"
        }

        if not required_keys.issubset(parsed.keys()):
            raise ValueError("Missing required keys in AI response")

        memory_update = parsed.get("memory_update") or {}
        memory_update = apply_score_clamp(brain, memory_update)

        parsed["memory_update"] = memory_update

        if supabase:
            new_brain = merge_brain(brain, memory_update)
            save_brain(student_id, new_brain)

        return parsed

    except Exception as e:
        # REAL ERROR — NO SILENT MASKING
        raise HTTPException(status_code=500, detail=str(e))
