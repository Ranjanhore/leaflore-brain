import os
import json
import logging
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client, Client

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)

# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(title="Leaflore NeuroAdaptive Brain v2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ENV CONFIG
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing Supabase credentials")

client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
You are Leaflore NeuroMentor — elite, calm, intelligent science teacher and life guide.

Return ONLY valid JSON with keys:
text
mode
next_action
micro_q
ui_hint
memory_update

No markdown.
No extra keys.
Always end with ONE micro-question.
"""

# ============================================================
# MEMORY UTILITIES (v2 Single Truth)
# ============================================================

def clamp(value: int, min_val=0, max_val=100) -> int:
    try:
        value = int(value)
    except:
        value = 50
    return max(min_val, min(value, max_val))

def default_brain() -> Dict[str, Any]:
    return {
        "mastery": {},
        "stress_score": 50,
        "confidence_score": 50,
        "personality_profile": {},
        "dream_map": {},
        "guardian_insights": {},
        "brain_data": {}
    }

def load_brain(student_id: str) -> Dict[str, Any]:
    res = (
        supabase.table("student_brain")
        .select("*")
        .eq("student_id", student_id)
        .limit(1)
        .execute()
    )

    if res.data and len(res.data) > 0:
        row = res.data[0]

        return {
            "mastery": row.get("mastery", {}),
            "stress_score": row.get("stress_score", 50),
            "confidence_score": row.get("confidence_score", 50),
            "personality_profile": row.get("personality_profile", {}),
            "dream_map": row.get("dream_map", {}),
            "guardian_insights": row.get("guardian_insights", {}),
            "brain_data": row.get("brain_data", {})
        }

    return default_brain()

def save_brain(student_id: str, brain: Dict[str, Any]) -> None:
    brain["stress_score"] = clamp(brain.get("stress_score", 50))
    brain["confidence_score"] = clamp(brain.get("confidence_score", 50))

    payload = {
        "student_id": student_id,
        "mastery": brain.get("mastery", {}),
        "stress_score": brain.get("stress_score"),
        "confidence_score": brain.get("confidence_score"),
        "personality_profile": brain.get("personality_profile", {}),
        "dream_map": brain.get("dream_map", {}),
        "guardian_insights": brain.get("guardian_insights", {}),
        "brain_data": brain.get("brain_data", {})
    }

    # IMPORTANT FIX
    supabase.table("student_brain") \
        .upsert(payload, on_conflict="student_id") \
        .execute()

def merge_brain(existing: Dict[str, Any], memory_update: Dict[str, Any]) -> Dict[str, Any]:
    merged = existing.copy()

    for key, value in memory_update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value

    merged["stress_score"] = clamp(merged.get("stress_score", 50))
    merged["confidence_score"] = clamp(merged.get("confidence_score", 50))

    return merged

# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {"ok": True, "version": "NeuroAdaptive Brain v2"}

@app.get("/health")
def health():
    return {"ok": True}

# ============================================================
# GET BRAIN
# ============================================================

@app.get("/student/{student_id}/brain")
def get_student_brain(student_id: str):
    brain = load_brain(student_id)
    return {"student_id": student_id, "brain": brain}

# ============================================================
# SET BRAIN (Manual Seed / Override)
# ============================================================

@app.post("/student/{student_id}/brain")
def set_student_brain(student_id: str, brain: Dict[str, Any]):
    full_brain = default_brain()
    full_brain.update(brain)
    save_brain(student_id, full_brain)
    return {"ok": True, "student_id": student_id}

# ============================================================
# AI RESPONSE
# ============================================================

@app.post("/respond")
def respond(req: BrainRequest):
    try:
        student_id = req.student_id or "demo"

        brain = load_brain(student_id)

        context = f"""
        Chapter: {req.chapter}
        Concept: {req.concept}
        Student Input: {req.student_input}
        Signals: {json.dumps(req.signals)}
        Brain: {json.dumps(brain)}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.6,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": context}
            ]
        )

        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)

        required = {
            "text",
            "mode",
            "next_action",
            "micro_q",
            "ui_hint",
            "memory_update"
        }

        if not required.issubset(parsed.keys()):
            raise ValueError("AI missing required keys")

        updated_brain = merge_brain(brain, parsed["memory_update"])
        save_brain(student_id, updated_brain)

        return parsed

    except Exception as e:
        logging.exception("AI ERROR")

        return {
            "text": "Let’s slow down and try again calmly.",
            "mode": "support",
            "next_action": "retry",
            "micro_q": "Which part feels unclear?",
            "ui_hint": "calm",
            "memory_update": {}
        }
