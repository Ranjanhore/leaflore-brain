# main.py — Leaflore Brain API (FastAPI + OpenAI + Supabase)
# Includes:
# - Supabase helpers (load/save/merge)
# - Strict JSON enforcement for model output
# - Confidence/stress auto-clamping with max ±10 per turn
# - Real error surfacing (DEBUG=1) + safe fallback in production
# - Global exception handler to reveal root cause when debugging

import os
import json
import traceback
from typing import Optional, Dict, Any, Set

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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

DEBUG = os.getenv("DEBUG", "0") == "1"

# ============================================================
# GLOBAL ERROR HANDLER (useful for debugging root 500s)
# ============================================================

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    if DEBUG:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(exc),
                "type": exc.__class__.__name__,
                "trace": traceback.format_exc(),
                "path": str(request.url),
            },
        )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error"},
    )

# ============================================================
# CLIENTS
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not OPENAI_API_KEY:
    # Fail fast: if OpenAI key missing, nothing works anyway.
    raise RuntimeError("Missing OPENAI_API_KEY env var")

# Supabase optional: app can run without it, but brain endpoints will error.
supabase: Optional[Client]
if not SUPABASE_URL or not SUPABASE_KEY:
    supabase = None
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

client = OpenAI(api_key=OPENAI_API_KEY)

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
# SYSTEM MASTER PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Leaflore NeuroMentor — an elite, warm, calm, highly intelligent science teacher, learning scientist, career mentor, guardian guide, and emotional support coach.

You are NOT a medical doctor.
You DO NOT diagnose, treat, label, or provide therapy.
No psychiatric claims.
All stress or confidence references are NON-CLINICAL learning indicators.

You support:
- Academic mastery
- Learning psychology
- Confidence building
- Stress management (non-clinical)
- Dream clarification
- Career mapping
- Parent guidance
- Healthy routines
- Communication and life balance

You teach clearly, patiently, intelligently, and age-appropriately.

====================================================
CRITICAL OUTPUT RULE
====================================================

Return ONLY valid JSON with EXACT keys:

text
mode
next_action
micro_q
ui_hint
memory_update

No markdown.
No extra keys.
No explanations outside JSON.

Always end with exactly ONE micro-question.

====================================================
SAFETY BOUNDARIES
====================================================

- No diagnosis.
- No therapy language.
- No medical claims.
- No psychiatric labels.
- If self-harm mentioned → calmly encourage contacting a trusted adult.

====================================================
MEMORY RULE
====================================================

Always update memory_update safely.
Confidence and stress change max ±10 per turn.
"""

# ============================================================
# HELPERS (SUPABASE)
# ============================================================

META_KEYS: Set[str] = {"id", "student_id", "updated_at"}

def _require_supabase():
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase is not configured (missing env vars).")

def load_brain(student_id: str) -> Dict[str, Any]:
    _require_supabase()
    res = (
        supabase.table("student_brain")
        .select("*")
        .eq("student_id", student_id)
        .limit(1)
        .execute()
    )
    if res.data and len(res.data) > 0:
        row = res.data[0] or {}
        # Return only brain fields (exclude metadata)
        return {k: v for k, v in row.items() if k not in META_KEYS and v is not None}
    return {}

def save_brain(student_id: str, brain_data: Dict[str, Any]) -> None:
    _require_supabase()
    if not isinstance(brain_data, dict):
        brain_data = {}
    payload = {"student_id": student_id, **brain_data}
    supabase.table("student_brain").upsert(payload).execute()

def merge_brain(existing: Dict[str, Any], memory_update: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(existing, dict):
        existing = {}
    if not isinstance(memory_update, dict):
        memory_update = {}
    merged = dict(existing)
    merged.update(memory_update)
    return merged

# ============================================================
# AUTO-CLAMPING LOGIC (confidence/stress)
# ============================================================

def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
    except Exception:
        v = default
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v

def clamp_scores(existing_brain: Dict[str, Any], memory_update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforces:
    - confidence_score and stress_score are ints between 0..100
    - each can change by max ±10 per turn (relative to existing brain)
    """
    if not isinstance(memory_update, dict):
        return {}

    out = dict(memory_update)

    # Current baseline
    base_conf = _clamp_int(existing_brain.get("confidence_score", 50), 0, 100, 50)
    base_stress = _clamp_int(existing_brain.get("stress_score", 40), 0, 100, 40)

    # Proposed
    if "confidence_score" in out:
        proposed = _clamp_int(out.get("confidence_score"), 0, 100, base_conf)
        lo, hi = base_conf - 10, base_conf + 10
        out["confidence_score"] = max(lo, min(hi, proposed))
    if "stress_score" in out:
        proposed = _clamp_int(out.get("stress_score"), 0, 100, base_stress)
        lo, hi = base_stress - 10, base_stress + 10
        out["stress_score"] = max(lo, min(hi, proposed))

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
# BRAIN STORAGE ENDPOINTS
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

        # 1) Load persisted brain if request brain is empty
        brain = req.brain or {}
        if (not brain) and supabase is not None:
            brain = load_brain(student_id)

        user_context = f"""
Chapter: {req.chapter}
Concept: {req.concept}
Student Input: {req.student_input}
Signals: {json.dumps(req.signals, ensure_ascii=False)}
Brain Memory (student profile): {json.dumps(brain, ensure_ascii=False)}
"""

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.6,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_context}
            ]
        )

        content = (response.choices[0].message.content or "").strip()

        # 2) Enforce strict JSON output
        parsed = json.loads(content)

        required_keys = {"text", "mode", "next_action", "micro_q", "ui_hint", "memory_update"}
        if not required_keys.issubset(parsed.keys()):
            raise ValueError(f"Missing required keys in AI response. Got keys: {list(parsed.keys())}")

        # 3) Persist memory_update into Supabase (merge into brain)
        memory_update = parsed.get("memory_update") or {}
        memory_update = clamp_scores(brain, memory_update)  # ✅ auto-clamping
        parsed["memory_update"] = memory_update             # keep output consistent with what is saved

        if supabase is not None:
            new_brain = merge_brain(brain, memory_update)
            save_brain(student_id, new_brain)

        return parsed

    except Exception as e:
        # If debugging, show the real error to fix faster.
        if DEBUG:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

        # Production-safe fallback (no internals leaked)
        return {
            "text": "I'm ready to help. Let’s take this step by step.",
            "mode": "support",
            "next_action": "retry",
            "micro_q": "Can you tell me what part feels unclear?",
            "ui_hint": "calm",
            "memory_update": {
                "confidence_score": 50,
                "stress_score": 40
            }
        }
