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
logger = logging.getLogger(__name__)

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

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    logger.warning("Supabase not configured. Brain persistence disabled.")

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

STRICT RULES:
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

No diagnosis. No therapy language. No medical claims.
Confidence & stress changes max ±10 per turn.
"""

# ============================================================
# SUPABASE HELPERS (JSONB brain_data column)
# ============================================================

def _require_supabase():
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase not configured")

def load_brain(student_id: str) -> Dict[str, Any]:
    _require_supabase()
    res = (
        supabase
        .table("student_brain")
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

    payload = {
        "student_id": student_id,
        "brain_data": brain_data
    }

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
# ROOT + HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "leaflore-brain",
        "message": "API is running"
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

        # Load existing brain if not provided
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

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            logger.error("Model returned invalid JSON")
            raise HTTPException(status_code=500, detail="Model returned invalid JSON")

        required_keys = {
            "text",
            "mode",
            "next_action",
            "micro_q",
            "ui_hint",
            "memory_update",
        }

        if not required_keys.issubset(parsed.keys()):
            raise HTTPException(status_code=500, detail="Model missing required keys")

        # Persist memory
        memory_update = parsed.get("memory_update") or {}
        if supabase:
            new_brain = merge_brain(brain, memory_update)
            save_brain(student_id, new_brain)

        return parsed

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Unhandled error in /respond")
        raise HTTPException(status_code=500, detail=str(e))
