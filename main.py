import os
import json
import logging
import traceback
from typing import Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

# Supabase (optional)
try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None
    Client = Any  # fallback typing


# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("leaflore-brain")


# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(title="Leaflore Brain API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL ERROR HANDLER (shows real error in JSON)
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log full traceback server-side
    tb = traceback.format_exc()
    logger.error("Unhandled error on %s %s: %s\n%s", request.method, request.url.path, str(exc), tb)

    # Return real error in response (dev-friendly).
    # If you want to hide details in production, set SHOW_ERRORS=false.
    show = os.getenv("SHOW_ERRORS", "true").lower() in ("1", "true", "yes")
    payload = {
        "ok": False,
        "error": str(exc) if show else "Internal Server Error",
        "where": tb if show else None,
    }
    return JSONResponse(status_code=500, content=payload)


# ============================================================
# CLIENTS / ENV
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var")

client = OpenAI(api_key=OPENAI_API_KEY)

supabase = None
if SUPABASE_URL and SUPABASE_KEY and create_client is not None:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase configured.")
    except Exception as e:
        supabase = None
        logger.warning("Supabase init failed: %s", e)
else:
    logger.info("Supabase not configured (missing env vars or supabase lib).")


# ============================================================
# REQUEST MODEL
# ============================================================

class BrainRequest(BaseModel):
    student_id: Optional[str] = Field(default="demo")
    chapter: Optional[str] = Field(default="")
    concept: Optional[str] = Field(default="")
    student_input: str
    signals: Optional[Dict[str, Any]] = Field(default_factory=dict)
    # Optional brain override (usually keep empty; server loads from Supabase)
    brain: Optional[Dict[str, Any]] = Field(default_factory=dict)


# ============================================================
# SYSTEM MASTER PROMPT (NeuroAdaptive Brain v2 - single truth)
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
"NeuroAdaptive Brain v2" MEMORY STRUCTURE (SINGLE TRUTH)
====================================================

memory_update MUST be a JSON object that can contain ANY of these keys only:

confidence_score: integer (0..100)
stress_score: integer (0..100)

mastery: object (json) where keys are concepts/skills and values are:
  level: integer (0..100)
  last_seen: string (optional)
  notes: string (optional)

personality_profile: object (json) e.g.
  learning_style, pace, preferred_examples, strengths, weaknesses

dream_map: object (json) e.g.
  goals, interests, aspirations, constraints

guardian_insights: object (json) e.g.
  parent_notes, home_routine, support_needed

brain_data: object (json) free-form but SAFE (no medical diagnosis)

If you are unsure, update only confidence_score and stress_score.

====================================================
MEMORY UPDATE RULES
====================================================

- Confidence and stress should change slowly: MAX ±10 per turn.
- Never output values below 0 or above 100.
- If you include mastery updates, keep them small and specific to the current concept.
- Be consistent with the student’s prior brain memory.

====================================================
SAFETY BOUNDARIES
====================================================

- No diagnosis.
- No therapy language.
- No medical claims.
- No psychiatric labels.
- If self-harm mentioned → calmly encourage contacting a trusted adult.
""".strip()


# ============================================================
# HELPERS (SUPABASE)
# ============================================================

META_KEYS = {"id", "student_id", "updated_at"}

def _require_supabase():
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase is not configured (missing env vars or init failed).")

def _is_dict(x: Any) -> bool:
    return isinstance(x, dict)

def _clamp_int(val: Any, lo: int, hi: int) -> int:
    try:
        v = int(val)
    except Exception:
        v = lo
    return max(lo, min(hi, v))

def _clamp_delta(new_val: int, old_val: int, max_delta: int = 10) -> int:
    # Enforce max ±max_delta change per turn
    if new_val > old_val + max_delta:
        return old_val + max_delta
    if new_val < old_val - max_delta:
        return old_val - max_delta
    return new_val

def _ensure_jsonb_object(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}

def load_brain(student_id: str) -> Dict[str, Any]:
    """
    Loads a student's brain row and returns only the brain fields (not metadata).
    IMPORTANT: We keep "single truth" in Supabase columns:
      mastery, stress_score, confidence_score, personality_profile, dream_map, guardian_insights, brain_data
    """
    _require_supabase()
    try:
        res = (
            supabase.table("student_brain")
            .select("*")
            .eq("student_id", student_id)
            .limit(1)
            .execute()
        )
        if res.data and len(res.data) > 0:
            row = res.data[0] or {}
            # Return only non-meta fields
            brain = {k: v for k, v in row.items() if k not in META_KEYS and v is not None}
            return brain
        return {}
    except Exception as e:
        logger.error("load_brain failed: %s", e)
        raise

def save_brain(student_id: str, brain_data: Dict[str, Any]) -> None:
    _require_supabase()
    if not isinstance(brain_data, dict):
        brain_data = {}
    payload = {"student_id": student_id, **brain_data}
    try:
        supabase.table("student_brain").upsert(payload).execute()
    except Exception as e:
        logger.error("save_brain failed: %s", e)
        raise

def merge_brain(existing: Dict[str, Any], memory_update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge memory_update into existing brain.
    - confidence_score + stress_score auto-clamped (0..100) and delta-limited (±10)
    - jsonb objects deep-merged at 1 level for known json fields
    """
    existing = existing if _is_dict(existing) else {}
    memory_update = memory_update if _is_dict(memory_update) else {}

    merged = dict(existing)

    # Ensure base structure exists (optional)
    merged.setdefault("mastery", {})
    merged.setdefault("personality_profile", {})
    merged.setdefault("dream_map", {})
    merged.setdefault("guardian_insights", {})
    merged.setdefault("brain_data", {})

    # Confidence / Stress clamp + delta limit
    old_conf = _clamp_int(merged.get("confidence_score", 50), 0, 100)
    old_stress = _clamp_int(merged.get("stress_score", 40), 0, 100)

    if "confidence_score" in memory_update:
        new_conf = _clamp_int(memory_update.get("confidence_score"), 0, 100)
        new_conf = _clamp_delta(new_conf, old_conf, 10)
        merged["confidence_score"] = new_conf
    else:
        merged["confidence_score"] = old_conf

    if "stress_score" in memory_update:
        new_stress = _clamp_int(memory_update.get("stress_score"), 0, 100)
        new_stress = _clamp_delta(new_stress, old_stress, 10)
        merged["stress_score"] = new_stress
    else:
        merged["stress_score"] = old_stress

    # Shallow merge json objects
    for key in ["mastery", "personality_profile", "dream_map", "guardian_insights", "brain_data"]:
        if key in memory_update and isinstance(memory_update[key], dict):
            base_obj = _ensure_jsonb_object(merged.get(key))
            upd_obj = _ensure_jsonb_object(memory_update.get(key))
            base_obj.update(upd_obj)
            merged[key] = base_obj

    return merged


# ============================================================
# ROUTES: ROOT & HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "leaflore-brain",
        "message": "API is running. Use /docs or POST /respond",
    }

@app.get("/health")
def health():
    return {"ok": True}


# ============================================================
# ROUTES: BRAIN STORAGE
# ============================================================

@app.get("/student/{student_id}/brain")
def get_student_brain(student_id: str):
    brain = load_brain(student_id)
    return {"student_id": student_id, "brain": brain}

@app.post("/student/{student_id}/brain")
def set_student_brain(student_id: str, brain: Dict[str, Any]):
    """
    Seed/overwrite brain data for a student.
    Provide only columns that exist:
      confidence_score, stress_score, mastery, personality_profile, dream_map, guardian_insights, brain_data
    """
    # Optional: normalize
    payload = brain if isinstance(brain, dict) else {}
    if "confidence_score" in payload:
        payload["confidence_score"] = _clamp_int(payload["confidence_score"], 0, 100)
    if "stress_score" in payload:
        payload["stress_score"] = _clamp_int(payload["stress_score"], 0, 100)
    for k in ["mastery", "personality_profile", "dream_map", "guardian_insights", "brain_data"]:
        if k in payload and not isinstance(payload[k], dict):
            payload[k] = {}
    save_brain(student_id, payload)
    return {"ok": True, "student_id": student_id}


# ============================================================
# ROUTE: MAIN AI ENDPOINT
# ============================================================

REQUIRED_KEYS = {"text", "mode", "next_action", "micro_q", "ui_hint", "memory_update"}

def _fallback_response(error_message: str, where: str = "") -> Dict[str, Any]:
    # Safe fallback in correct schema
    return {
        "text": "I’m ready to help. Let’s take this step by step.",
        "mode": "support",
        "next_action": "retry",
        "micro_q": "Can you tell me exactly what part feels unclear?",
        "ui_hint": "calm",
        "memory_update": {
            "confidence_score": 50,
            "stress_score": 40,
            "brain_data": {
                "last_error": error_message[:300],
                "where": where[:500],
            },
        },
    }

def _parse_ai_json(content: str) -> Dict[str, Any]:
    # Try strict parse, then minimal repair for common cases (extra text before/after JSON)
    try:
        return json.loads(content)
    except Exception:
        # Try to extract JSON object substring
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = content[start : end + 1]
            return json.loads(snippet)
        raise

@app.post("/respond")
def respond(req: BrainRequest):
    student_id = (req.student_id or "demo").strip() or "demo"

    # 1) Load persisted brain if request brain is empty
    brain = req.brain or {}
    if (not brain) and supabase is not None:
        brain = load_brain(student_id)

    user_context = f"""
Chapter: {req.chapter}
Concept: {req.concept}
Student Input: {req.student_input}
Signals: {json.dumps(req.signals or {}, ensure_ascii=False)}
Brain Memory (student profile): {json.dumps(brain or {}, ensure_ascii=False)}
""".strip()

    try:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.6,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_context},
            ],
        )

        content = (response.choices[0].message.content or "").strip()
        parsed = _parse_ai_json(content)

        # Validate schema
        if not isinstance(parsed, dict) or not REQUIRED_KEYS.issubset(parsed.keys()):
            raise ValueError(f"AI response missing required keys. Got keys: {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}")

        if not isinstance(parsed.get("memory_update"), dict):
            parsed["memory_update"] = {}

        # 2) Persist memory_update into Supabase (merge into brain)
        if supabase is not None:
            new_brain = merge_brain(brain or {}, parsed.get("memory_update") or {})
            save_brain(student_id, new_brain)

        return parsed

    except HTTPException:
        # re-raise FastAPI errors as-is
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("respond() failed: %s\n%s", str(e), tb)

        # If you want to expose error details to client in this endpoint too:
        show = os.getenv("SHOW_ERRORS", "true").lower() in ("1", "true", "yes")
        msg = str(e)
        where = tb if show else ""

        return _fallback_response(error_message=msg, where=where)
