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
# ENV + CLIENTS
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

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
You are Leaflore NeuroMentor — elite, calm, intelligent science mentor.

Return ONLY valid JSON with EXACT keys:

text
mode
next_action
micro_q
ui_hint
memory_update

No markdown.
No extra keys.
One micro-question only.

Confidence and stress changes max ±10 per turn.
Never exceed 0–100.
No therapy or diagnosis.
"""


# ============================================================
# SUPABASE HELPERS
# ============================================================

def _require_supabase():
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase not configured")

META_KEYS = {"id", "student_id", "updated_at"}


def load_brain(student_id: str) -> Dict[str, Any]:
    _require_supabase()

    res = (
        supabase.table("student_brain")
        .select("*")
        .eq("student_id", student_id)
        .limit(1)
        .execute()
    )

    if res.data:
        row = res.data[0] or {}
        return {
            "confidence_score": row.get("confidence_score"),
            "stress_score": row.get("stress_score"),
            "mastery": row.get("mastery", {}),
            "personality_profile": row.get("personality_profile", {}),
            "dream_map": row.get("dream_map", {}),
            "guardian_insights": row.get("guardian_insights", {}),
            "brain_data": row.get("brain_data", {}),
        }

    return {}


def save_brain(student_id: str, brain_data: Dict[str, Any]) -> None:
    _require_supabase()

    payload = {
        "student_id": student_id,
        "confidence_score": brain_data.get("confidence_score"),
        "stress_score": brain_data.get("stress_score"),
        "mastery": brain_data.get("mastery", {}),
        "personality_profile": brain_data.get("personality_profile", {}),
        "dream_map": brain_data.get("dream_map", {}),
        "guardian_insights": brain_data.get("guardian_insights", {}),
        "brain_data": brain_data,
    }

    supabase.table("student_brain").upsert(payload).execute()


def merge_brain(existing: Dict[str, Any], memory_update: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(existing, dict):
        existing = {}

    if not isinstance(memory_update, dict):
        memory_update = {}

    merged = dict(existing)
    merged.update(memory_update)

    # 🔒 Auto clamp
    if "confidence_score" in merged:
        merged["confidence_score"] = max(0, min(100, merged["confidence_score"]))

    if "stress_score" in merged:
        merged["stress_score"] = max(0, min(100, merged["stress_score"]))

    return merged


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"ok": True, "service": "leaflore-brain"}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/student/{student_id}/brain")
def get_student_brain(student_id: str):
    brain = load_brain(student_id) if supabase else {}
    return {"student_id": student_id, "brain": brain}


@app.post("/student/{student_id}/brain")
def set_student_brain(student_id: str, brain: Dict[str, Any]):
    save_brain(student_id, brain)
    return {"ok": True}


# ============================================================
# MAIN AI ENDPOINT
# ============================================================

@app.post("/respond")
def respond(req: BrainRequest):

    try:
        student_id = (req.student_id or "demo").strip()

        # Load brain from DB if not passed
        brain = req.brain or {}
        if not brain and supabase:
            brain = load_brain(student_id)

        user_context = f"""
Chapter: {req.chapter}
Concept: {req.concept}
Student Input: {req.student_input}
Signals: {json.dumps(req.signals)}
Brain Memory: {json.dumps(brain)}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.6,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_context},
            ],
        )

        content = (response.choices[0].message.content or "").strip()

        # Strict JSON parsing
        parsed = json.loads(content)

        required = {
            "text",
            "mode",
            "next_action",
            "micro_q",
            "ui_hint",
            "memory_update",
        }

        if not required.issubset(parsed.keys()):
            raise ValueError("AI output missing required keys")

        memory_update = parsed.get("memory_update") or {}

        if supabase:
            updated_brain = merge_brain(brain, memory_update)
            save_brain(student_id, updated_brain)

        return parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
