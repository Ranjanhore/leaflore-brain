import os
import json
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI
from supabase import create_client, Client

# ============================================================
# CONFIG
# ============================================================

APP_NAME = "Leaflore NeuroAdaptive Brain v2 (All Features)"
DEBUG = os.getenv("DEBUG", "0") == "1"

logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing Supabase credentials (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)")

client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ok for dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# GLOBAL ERROR HANDLER (REAL ERROR WHEN DEBUG=1)
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    logging.exception(f"[{req_id}] Unhandled error on {request.method} {request.url}")

    if DEBUG:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": {
                    "message": str(exc),
                    "type": exc.__class__.__name__,
                },
                "where": "global_exception_handler",
                "request_id": req_id,
            },
        )

    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": {"message": "Internal Server Error"},
            "request_id": req_id,
        },
    )

# ============================================================
# REQUEST MODELS
# ============================================================

class BrainRequest(BaseModel):
    student_id: Optional[str] = "demo"
    chapter: Optional[str] = ""
    concept: Optional[str] = ""
    student_input: str
    signals: Optional[Dict[str, Any]] = {}
    brain: Optional[Dict[str, Any]] = {}  # optional override (rare)

# ============================================================
# SYSTEM PROMPT (WITH DELTA RULE)
# ============================================================

SYSTEM_PROMPT = """
You are Leaflore NeuroMentor — elite, warm, calm, highly intelligent science teacher and learning coach.

CRITICAL OUTPUT RULE:
Return ONLY valid JSON with EXACT keys:
text
mode
next_action
micro_q
ui_hint
memory_update

No markdown. No extra keys. Always end with exactly ONE micro-question.

MEMORY RULES:
- memory_update is a partial update (patch) to the student's brain.
- If you update confidence_score or stress_score, change should be SMALL (think ±10 max).
- Prefer updating mastery for the given concept/chapter.
- Keep everything non-clinical (learning indicators only).
"""

# ============================================================
# BRAIN STRUCTURE (SINGLE TRUTH)
# ============================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def clamp_int(v: Any, lo: int = 0, hi: int = 100, default: int = 50) -> int:
    try:
        v = int(v)
    except Exception:
        v = default
    return max(lo, min(hi, v))

def clamp_delta(prev: int, new: int, max_step: int = 10) -> int:
    """Limit change per turn to ±max_step."""
    if new > prev + max_step:
        return prev + max_step
    if new < prev - max_step:
        return prev - max_step
    return new

def default_brain() -> Dict[str, Any]:
    return {
        "mastery": {},  # e.g. mastery["Photosynthesis::Chlorophyll"] = {score, attempts, last_seen, streak}
        "stress_score": 50,
        "confidence_score": 50,
        "personality_profile": {},
        "dream_map": {},
        "guardian_insights": {},
        "brain_data": {
            "version": "v2",
            "last_interaction_at": None,
            "emotion_history": [],  # list of {ts, emotion, engagement, chapter, concept}
            "trend": {
                "stress_7": [],
                "confidence_7": [],
                "neuro_7": [],
            },
            "last_ai": {
                "mode": None,
                "next_action": None,
            },
            "debug": {
                "last_merge_keys": [],
                "last_request_id": None,
            },
        },
    }

def normalize_brain(brain: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all required keys exist and types are sane."""
    base = default_brain()

    if not isinstance(brain, dict):
        return base

    for k in ["mastery", "personality_profile", "dream_map", "guardian_insights"]:
        if not isinstance(brain.get(k), dict):
            brain[k] = {}

    if not isinstance(brain.get("brain_data"), dict):
        brain["brain_data"] = {}

    # merge brain_data defaults
    bd = base["brain_data"]
    bd_in = brain.get("brain_data", {})
    if not isinstance(bd_in, dict):
        bd_in = {}
    bd.update(bd_in)
    brain["brain_data"] = bd

    brain["stress_score"] = clamp_int(brain.get("stress_score", 50))
    brain["confidence_score"] = clamp_int(brain.get("confidence_score", 50))

    # ensure lists
    if not isinstance(brain["brain_data"].get("emotion_history"), list):
        brain["brain_data"]["emotion_history"] = []
    if not isinstance(brain["brain_data"].get("trend"), dict):
        brain["brain_data"]["trend"] = {"stress_7": [], "confidence_7": [], "neuro_7": []}

    for key in ["stress_7", "confidence_7", "neuro_7"]:
        if not isinstance(brain["brain_data"]["trend"].get(key), list):
            brain["brain_data"]["trend"][key] = []

    return {**base, **brain}

# ============================================================
# SUPABASE IO (matches your table columns)
# columns:
# id uuid, student_id text UNIQUE,
# mastery jsonb, stress_score int, confidence_score int,
# personality_profile jsonb, dream_map jsonb, guardian_insights jsonb,
# updated_at timestamp, brain_data jsonb
# ============================================================

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
        brain = {
            "mastery": row.get("mastery", {}) or {},
            "stress_score": row.get("stress_score", 50),
            "confidence_score": row.get("confidence_score", 50),
            "personality_profile": row.get("personality_profile", {}) or {},
            "dream_map": row.get("dream_map", {}) or {},
            "guardian_insights": row.get("guardian_insights", {}) or {},
            "brain_data": row.get("brain_data", {}) or {},
        }
        return normalize_brain(brain)

    return normalize_brain(default_brain())

def save_brain(student_id: str, brain: Dict[str, Any]) -> None:
    brain = normalize_brain(brain)

    payload = {
        "student_id": student_id,
        "mastery": brain["mastery"],
        "stress_score": brain["stress_score"],
        "confidence_score": brain["confidence_score"],
        "personality_profile": brain["personality_profile"],
        "dream_map": brain["dream_map"],
        "guardian_insights": brain["guardian_insights"],
        "brain_data": brain["brain_data"],
    }

    # IMPORTANT: prevents duplicate key error
    supabase.table("student_brain").upsert(payload, on_conflict="student_id").execute()

# ============================================================
# ANALYTICS / DERIVED METRICS
# ============================================================

def mastery_key(chapter: str, concept: str) -> str:
    chapter = (chapter or "").strip() or "General"
    concept = (concept or "").strip() or "Concept"
    return f"{chapter}::{concept}"

def mastery_avg(mastery: Dict[str, Any]) -> int:
    if not isinstance(mastery, dict) or not mastery:
        return 0
    scores = []
    for _, v in mastery.items():
        if isinstance(v, dict) and "score" in v:
            scores.append(clamp_int(v.get("score", 0), 0, 100, 0))
    if not scores:
        return 0
    return int(sum(scores) / len(scores))

def neuro_score(brain: Dict[str, Any]) -> int:
    """Simple composite metric."""
    conf = clamp_int(brain.get("confidence_score", 50))
    stress = clamp_int(brain.get("stress_score", 50))
    mavg = mastery_avg(brain.get("mastery", {}))
    # higher is better; keep 0..100
    # conf ↑, mastery ↑, stress ↓
    raw = int((conf + mavg + (100 - stress)) / 3)
    return clamp_int(raw, 0, 100, 50)

def push_trend(brain: Dict[str, Any]) -> None:
    bd = brain["brain_data"]
    tr = bd["trend"]
    tr["stress_7"].append(brain["stress_score"])
    tr["confidence_7"].append(brain["confidence_score"])
    tr["neuro_7"].append(neuro_score(brain))
    # keep last 7
    for k in ["stress_7", "confidence_7", "neuro_7"]:
        tr[k] = tr[k][-7:]
    bd["trend"] = tr
    brain["brain_data"] = bd

def add_emotion_event(brain: Dict[str, Any], *, chapter: str, concept: str, signals: Dict[str, Any]) -> None:
    bd = brain["brain_data"]
    hist: List[Dict[str, Any]] = bd.get("emotion_history", [])
    if not isinstance(hist, list):
        hist = []

    emotion = (signals or {}).get("emotion")
    engagement = (signals or {}).get("engagement")
    if emotion or engagement:
        hist.append({
            "ts": now_iso(),
            "emotion": emotion,
            "engagement": engagement,
            "chapter": chapter,
            "concept": concept,
        })
        hist = hist[-30:]  # keep last 30 events

    bd["emotion_history"] = hist
    brain["brain_data"] = bd

# ============================================================
# MASTERY AUTO-SCALING (SERVER-SIDE)
# ============================================================

def mastery_update_auto(
    brain: Dict[str, Any],
    chapter: str,
    concept: str,
    signals: Dict[str, Any],
    ai_mode: str,
    next_action: str,
) -> None:
    """
    Heuristic:
    - confused / low engagement => small mastery decrease or no gain
    - success / proceed => small gain
    - always tiny steps to avoid jumps
    """
    mk = mastery_key(chapter, concept)
    mastery = brain.get("mastery", {})
    if not isinstance(mastery, dict):
        mastery = {}

    node = mastery.get(mk, {})
    if not isinstance(node, dict):
        node = {}

    prev_score = clamp_int(node.get("score", 30), 0, 100, 30)
    attempts = int(node.get("attempts", 0) or 0)
    streak = int(node.get("streak", 0) or 0)

    emotion = (signals or {}).get("emotion", "") or ""
    engagement = (signals or {}).get("engagement", "") or ""

    delta = 0

    # negative signals
    if str(emotion).lower() in ["confused", "frustrated", "stressed"]:
        delta -= 2
        streak = max(0, streak - 1)

    if str(engagement).lower() in ["low"]:
        delta -= 1

    # positive signals: if AI says proceed/next or mode tutor
    if next_action in ["next", "continue", "proceed"]:
        delta += 2
        streak += 1

    if ai_mode in ["teach", "coach", "tutor", "support"] and next_action in ["retry"]:
        # guided retry can still be a small gain (learning happens)
        delta += 1

    # keep very small changes
    delta = max(-3, min(3, delta))

    new_score = clamp_int(prev_score + delta, 0, 100, prev_score)

    node.update({
        "score": new_score,
        "attempts": attempts + 1,
        "streak": streak,
        "last_seen": now_iso(),
    })

    mastery[mk] = node
    brain["mastery"] = mastery

# ============================================================
# MERGE (PATCH) + SCORE DELTA LIMITS + DIFF LOGGING
# ============================================================

def deep_merge_dict(dst: Dict[str, Any], patch: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    keys_touched: List[str] = []

    def _merge(a: Any, b: Any, path: str) -> Any:
        nonlocal keys_touched
        if isinstance(a, dict) and isinstance(b, dict):
            out = dict(a)
            for k, v in b.items():
                out[k] = _merge(out.get(k), v, f"{path}.{k}" if path else k)
            return out
        else:
            keys_touched.append(path)
            return b

    return _merge(dst, patch, ""), keys_touched

def apply_score_rules(existing: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports:
    - confidence_score / stress_score as absolute (but delta limited ±10)
    - confidence_delta / stress_delta as delta (also limited ±10)
    """
    out = dict(patch) if isinstance(patch, dict) else {}

    prev_conf = clamp_int(existing.get("confidence_score", 50))
    prev_stress = clamp_int(existing.get("stress_score", 50))

    # delta fields
    if "confidence_delta" in out:
        try:
            d = int(out.pop("confidence_delta"))
        except:
            d = 0
        d = max(-10, min(10, d))
        out["confidence_score"] = clamp_int(prev_conf + d)

    if "stress_delta" in out:
        try:
            d = int(out.pop("stress_delta"))
        except:
            d = 0
        d = max(-10, min(10, d))
        out["stress_score"] = clamp_int(prev_stress + d)

    # absolute fields with delta clamp
    if "confidence_score" in out:
        out["confidence_score"] = clamp_int(out["confidence_score"])
        out["confidence_score"] = clamp_delta(prev_conf, out["confidence_score"], 10)

    if "stress_score" in out:
        out["stress_score"] = clamp_int(out["stress_score"])
        out["stress_score"] = clamp_delta(prev_stress, out["stress_score"], 10)

    return out

def merge_brain(existing: Dict[str, Any], memory_update: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    existing = normalize_brain(existing)

    if not isinstance(memory_update, dict):
        memory_update = {}

    memory_update = apply_score_rules(existing, memory_update)

    before = json.dumps(existing, sort_keys=True)
    merged, touched = deep_merge_dict(existing, memory_update)

    merged = normalize_brain(merged)
    merged["brain_data"]["last_interaction_at"] = now_iso()
    merged["brain_data"]["debug"]["last_merge_keys"] = touched[-50:]
    merged["brain_data"]["debug"]["last_request_id"] = request_id

    # log a short diff signal (not full content)
    after = json.dumps(merged, sort_keys=True)
    if before != after:
        logging.info(f"[{request_id}] brain_updated keys={touched[-20:]}")

    return merged

# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {"ok": True, "service": "leaflore-brain", "version": "v2-all", "debug": DEBUG}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/student/{student_id}/brain")
def get_student_brain(student_id: str):
    brain = load_brain(student_id)
    return {"student_id": student_id, "brain": brain}

@app.post("/student/{student_id}/brain")
def set_student_brain(student_id: str, brain_patch: Dict[str, Any]):
    """
    Seed/Override brain (v2 single-truth).
    Paste ONLY the brain object here (not nested).
    """
    request_id = str(uuid.uuid4())

    current = load_brain(student_id)
    brain_patch = brain_patch if isinstance(brain_patch, dict) else {}
    # treat as a patch (safe)
    merged = merge_brain(current, brain_patch, request_id)
    save_brain(student_id, merged)

    return {"ok": True, "student_id": student_id, "brain": merged}

@app.get("/student/{student_id}/analytics")
def get_student_analytics(student_id: str):
    brain = load_brain(student_id)
    return {
        "student_id": student_id,
        "neuro_score": neuro_score(brain),
        "confidence_score": brain["confidence_score"],
        "stress_score": brain["stress_score"],
        "mastery_avg": mastery_avg(brain["mastery"]),
        "mastery_count": len(brain["mastery"]),
        "last_interaction_at": brain["brain_data"].get("last_interaction_at"),
        "trend": brain["brain_data"].get("trend", {}),
        "recent_emotions": (brain["brain_data"].get("emotion_history") or [])[-5:],
    }

# ============================================================
# MAIN AI ENDPOINT
# ============================================================

@app.post("/respond")
def respond(req: BrainRequest):
    request_id = str(uuid.uuid4())
    t0 = time.time()

    student_id = (req.student_id or "demo").strip()

    # 1) Load brain (single truth)
    brain = load_brain(student_id)

    # 2) Optional override: if req.brain provided, treat as patch (rare)
    if isinstance(req.brain, dict) and req.brain:
        brain = merge_brain(brain, req.brain, request_id)

    # 3) Track signals (emotion history)
    add_emotion_event(brain, chapter=req.chapter, concept=req.concept, signals=req.signals)

    user_context = {
        "chapter": req.chapter,
        "concept": req.concept,
        "student_input": req.student_input,
        "signals": req.signals or {},
        "brain": brain,  # send current state to model
        "request_id": request_id,
    }

    # 4) Call OpenAI with JSON enforcement (and fallback)
    try:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.6,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_context, ensure_ascii=False)},
                ],
            )
        except TypeError:
            # older client fallback (no response_format)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.6,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(user_context, ensure_ascii=False)},
                ],
            )

        content = (response.choices[0].message.content or "").strip()
        parsed = json.loads(content)

        required = {"text", "mode", "next_action", "micro_q", "ui_hint", "memory_update"}
        if not required.issubset(parsed.keys()):
            raise ValueError(f"AI missing keys: {sorted(list(required - set(parsed.keys())))}")

        # 5) Server-side mastery auto-scaling (always)
        ai_mode = str(parsed.get("mode", "") or "")
        next_action = str(parsed.get("next_action", "") or "")
        mastery_update_auto(brain, req.chapter, req.concept, req.signals or {}, ai_mode, next_action)

        # 6) Merge model memory_update (with delta clamp) into brain
        memory_update = parsed.get("memory_update") or {}
        merged = merge_brain(brain, memory_update, request_id)

        # 7) Update AI last fields + trend
        merged["brain_data"]["last_ai"]["mode"] = ai_mode
        merged["brain_data"]["last_ai"]["next_action"] = next_action
        push_trend(merged)

        # 8) Persist
        save_brain(student_id, merged)

        # 9) Optional: attach debug/analytics to help Lovable integration (safe to remove)
        if DEBUG:
            parsed["_debug"] = {
                "request_id": request_id,
                "latency_ms": int((time.time() - t0) * 1000),
                "neuro_score": neuro_score(merged),
                "mastery_avg": mastery_avg(merged["mastery"]),
            }

        return parsed

    except Exception as e:
        logging.exception(f"[{request_id}] /respond failed")

        if DEBUG:
            raise HTTPException(
                status_code=500,
                detail={
                    "ok": False,
                    "error": {"message": str(e), "type": e.__class__.__name__},
                    "where": "/respond",
                    "request_id": request_id,
                },
            )

        # safe fallback (still valid schema)
        return {
            "text": "Let’s slow down and try again calmly.",
            "mode": "support",
            "next_action": "retry",
            "micro_q": "Which part feels unclear?",
            "ui_hint": "calm",
            "memory_update": {},
        }
