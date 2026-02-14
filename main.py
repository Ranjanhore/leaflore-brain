import os
import json
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from openai import OpenAI
from supabase import create_client, Client

# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("leaflore-brain")

# ============================================================
# APP (MUST BE TOP-LEVEL FOR Render: uvicorn main:app)
# ============================================================

app = FastAPI(title="Leaflore Brain API", version="prod-1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# DEBUG: Supabase Environment Check
# ================================

@app.get("/debug/supabase")
def debug_supabase():
    return {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SERVICE_ROLE_KEY_LENGTH": len(os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")),
    }


@app.get("/debug/columns")
def debug_columns():
    from supabase import create_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    sb = create_client(url, key)

    res = sb.table("student_brain").select("mastery_rollup").limit(1).execute()

    return {
        "data": res.data,
        "error": str(res.error) if res.error else None
    }


# ============================================================
# ENV + CLIENTS
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Optional API protection (recommended when called from Lovable edge proxy)
BRAIN_API_KEY = os.getenv("BRAIN_API_KEY")  # if set -> require x-api-key header

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var")

client = OpenAI(api_key=OPENAI_API_KEY)

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    logger.warning("Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to enable DB features.")

# ============================================================
# SECURITY: x-api-key (if BRAIN_API_KEY is configured)
# ============================================================

@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if not BRAIN_API_KEY:
        return await call_next(request)

    # allow health/root without key for uptime checks (optional)
    if request.url.path in ["/", "/health"]:
        return await call_next(request)

    key = request.headers.get("x-api-key")
    if not key or key != BRAIN_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: invalid or missing x-api-key")
    return await call_next(request)

# ============================================================
# REQUEST MODELS
# ============================================================

class BrainRequest(BaseModel):
    student_id: Optional[str] = "demo"

    # Curriculum routing
    board: Optional[str] = "ICSE"
    grade: Optional[int] = 6
    subject: Optional[str] = "Science"
    chapter: Optional[str] = ""
    concept: Optional[str] = ""
    language: Optional[str] = "en"  # request language (en/hi/hinglish or variants)

    # Conversation input
    student_input: str = Field(..., min_length=1)

    # Signals: actor/emotion/engagement/event/student_name/preferred_language/...
    signals: Optional[Dict[str, Any]] = {}

    # Optional brain override
    brain: Optional[Dict[str, Any]] = {}


class QuizStartRequest(BaseModel):
    board: str = "ICSE"
    grade: int = 6
    subject: str = "Science"
    chapter: str
    concept: str
    language: str = "en"
    student_ids: List[str] = []
    representative_student_id: Optional[str] = None
    student_level: Optional[int] = None  # 1..5 optional override


class QuizAnswerRequest(BaseModel):
    session_id: str
    student_id: str
    answer: str

# ============================================================
# SYSTEM PROMPT (loaded from env if provided)
# ============================================================

DEFAULT_SYSTEM_PROMPT = """
You are Leaflore NeuroMentor — an elite, warm, calm, highly intelligent teacher, learning scientist, career mentor, guardian guide, and emotional support coach.

You are NOT a medical doctor.
You DO NOT diagnose, treat, label, or provide therapy.
No psychiatric claims.
All stress/confidence references are NON-CLINICAL learning indicators only.

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

====================================================
CRITICAL OUTPUT RULE (STRICT)
====================================================
Return ONLY valid JSON with EXACT keys (and nothing else):
text
mode
next_action
micro_q
ui_hint
memory_update

No markdown.
No extra keys.
No explanations outside JSON.
No trailing commas.
Always end with EXACTLY ONE micro-question (placed in micro_q).
The "text" can be multiple short lines, but keep it concise.

====================================================
INPUTS YOU WILL RECEIVE (READ-ONLY)
====================================================
You will receive a JSON user payload containing:
- board, grade, subject, chapter, concept
- student_input
- signals (actor, emotion, event, class_end, student_name, preferred_language, language, etc.)
- brain (student profile + telemetry + mastery)
- chunks (curriculum content array)

Treat "chunks" as the primary source of truth when present.

====================================================
CURRICULUM CHUNKS (MANDATORY)
====================================================
If chunks are present:
- You MUST use at least ONE fact/line that is clearly derived from the chunks.
- You MUST stay aligned to the chapter + concept.
- Do NOT invent textbook facts that conflict with chunks.
- Keep explanations in small steps.

Chunk priority order:
1) explain
2) misconception
3) example
4) recap
5) quiz (if provided separately)

If chunks are missing:
- Teach generally, simply, and safely.
- Ask ONE micro-question to confirm understanding.

====================================================
ACTOR ROUTING (STUDENT VS PARENT CONTROL)
====================================================
You will receive signals.actor in {"student","parent","other","unknown"}.

If actor == "parent":
- mode = "parent_support"
- Do NOT quiz the parent like a student.
- micro_q must be a parent question.

If actor == "student":
- mode = "teach" or "quiz"
- Ask ONE student micro-question.

If actor in {"other","unknown"}:
- mode="support"
- next_action="clarify_actor"

====================================================
AUTO SCREEN HEADLINE GENERATION (ui_hint STRING ONLY)
====================================================
ui_hint must be a STRING:
"headline=... | sub=... | meta=... | badge1=... | badge2=... | badge3=..."

====================================================
MEMORY UPDATE RULE
====================================================
You MUST set memory_update every turn.
Never change confidence_score or stress_score by more than ±10 per turn.
""".strip()

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT).strip()

# ============================================================
# UTILS
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(value)
    except Exception:
        return default
    return max(lo, min(hi, v))

def clamp_delta(delta: Any, limit: int = 10) -> int:
    try:
        d = int(delta)
    except Exception:
        d = 0
    return max(-limit, min(limit, d))

def get_actor(signals: Dict[str, Any]) -> str:
    a = (signals or {}).get("actor") or "unknown"
    a = str(a).strip().lower()
    if a not in ["student", "parent", "other", "unknown"]:
        return "unknown"
    return a

def concept_key(chapter: str, concept: str) -> str:
    ch = (chapter or "").strip()
    co = (concept or "").strip()
    return f"{ch}::{co}".strip(":")

# ============================================================
# LANGUAGE (en / hi / hinglish)
# ============================================================

ALLOWED_LANGS = {"en", "hi", "hinglish"}

def normalize_language(value: Any) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()

    if v in {"english", "eng", "en-us", "en_in", "en-gb"}:
        return "en"
    if v in {"hindi", "hin", "hi-in"}:
        return "hi"
    if v in {"hinglish", "hi-en", "hin-eng", "hindi+english"}:
        return "hinglish"

    if v in ALLOWED_LANGS:
        return v
    return None

def resolve_preferred_language(req_language: Any, signals: Dict[str, Any], brain: Dict[str, Any]) -> str:
    signals = signals or {}
    brain = brain or {}
    raw = (
        signals.get("preferred_language")
        or signals.get("language")
        or req_language
        or brain.get("preferred_language")
    )
    return normalize_language(raw) or "hinglish"

def resolve_teaching_language(req_language: Any, signals: Dict[str, Any], brain: Dict[str, Any]) -> str:
    preferred = resolve_preferred_language(req_language, signals, brain)
    if preferred == "hi":
        return "hinglish"
    return preferred

# ============================================================
# SUPABASE HELPERS
# ============================================================

META_KEYS = {"id", "student_id", "updated_at", "created_at"}

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
        return {k: v for k, v in row.items() if k not in META_KEYS and v is not None}
    return {}

def save_brain(student_id: str, brain_data: Dict[str, Any]) -> None:
    _require_supabase()
    if not isinstance(brain_data, dict):
        brain_data = {}
    payload = {"student_id": student_id, **brain_data}
    supabase.table("student_brain").upsert(payload).execute()

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(a, dict):
        a = {}
    if not isinstance(b, dict):
        b = {}
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

# ============================================================
# TURN-LOCK (SOFT GATE)
# ============================================================

def ensure_turnlock(brain: Dict[str, Any]) -> Dict[str, Any]:
    brain = brain or {}
    bd = brain.get("brain_data") or {}
    tel = bd.get("telemetry") or {}
    lock = tel.get("turn_lock") or {}

    if not isinstance(lock, dict):
        lock = {}

    lock.setdefault("enabled", False)
    lock.setdefault("expected_actor", None)
    lock.setdefault("turn_id", None)
    lock.setdefault("purpose", None)
    lock.setdefault("created_at", None)

    tel["turn_lock"] = lock
    bd["telemetry"] = tel
    brain["brain_data"] = bd
    return brain

def set_turn_lock(brain: Dict[str, Any], expected_actor: str, purpose: str = "micro_q") -> Dict[str, Any]:
    brain = ensure_turnlock(brain)
    tel = brain["brain_data"]["telemetry"]
    lock = tel["turn_lock"]
    lock["enabled"] = True
    lock["expected_actor"] = expected_actor
    lock["purpose"] = purpose
    lock["turn_id"] = f"t_{uuid.uuid4().hex[:10]}"
    lock["created_at"] = utc_now_iso()
    tel["turn_lock"] = lock
    brain["brain_data"]["telemetry"] = tel
    return brain

def clear_turn_lock(brain: Dict[str, Any]) -> Dict[str, Any]:
    brain = ensure_turnlock(brain)
    tel = brain["brain_data"]["telemetry"]
    lock = tel["turn_lock"]
    lock["enabled"] = False
    lock["expected_actor"] = None
    lock["purpose"] = None
    lock["turn_id"] = None
    lock["created_at"] = None
    tel["turn_lock"] = lock
    brain["brain_data"]["telemetry"] = tel
    return brain

def is_actor_allowed(brain: Dict[str, Any], incoming_actor: str) -> bool:
    brain = ensure_turnlock(brain)
    lock = brain["brain_data"]["telemetry"]["turn_lock"]
    if not lock.get("enabled"):
        return True
    expected = (lock.get("expected_actor") or "").lower()
    if not expected:
        return True
    return incoming_actor == expected

# ============================================================
# CURRICULUM CHUNKS
# ============================================================

def fetch_explain_chunks(board: str, grade: int, subject: str, chapter: str, concept: str, language: str, limit: int = 6):
    _require_supabase()
    res = (
        supabase.table("curriculum_chunks")
        .select("id,chunk_type,difficulty,chunk_text,tags")
        .eq("board", board)
        .eq("grade", int(grade))
        .eq("subject", subject)
        .eq("chapter", chapter)
        .eq("concept", concept)
        .eq("language", language)
        .in_("chunk_type", ["explain", "example", "misconception", "recap"])
        .order("difficulty", desc=False)
        .limit(limit)
        .execute()
    )
    return res.data or []

def fetch_quiz_chunk(board: str, grade: int, subject: str, chapter: str, concept: str, language: str, difficulty: int):
    _require_supabase()
    res = (
        supabase.table("curriculum_chunks")
        .select("*")
        .eq("board", board)
        .eq("grade", int(grade))
        .eq("subject", subject)
        .eq("chapter", chapter)
        .eq("concept", concept)
        .eq("language", language)
        .eq("chunk_type", "quiz")
        .eq("difficulty", int(difficulty))
        .limit(1)
        .execute()
    )
    if res.data:
        return res.data[0]

    for d in [difficulty - 1, difficulty + 1, difficulty - 2, difficulty + 2]:
        if d < 1 or d > 5:
            continue
        res2 = (
            supabase.table("curriculum_chunks")
            .select("*")
            .eq("board", board)
            .eq("grade", int(grade))
            .eq("subject", subject)
            .eq("chapter", chapter)
            .eq("concept", concept)
            .eq("language", language)
            .eq("chunk_type", "quiz")
            .eq("difficulty", int(d))
            .limit(1)
            .execute()
        )
        if res2.data:
            return res2.data[0]
    return None

def save_quiz_chunk(
    board: str, grade: int, subject: str, chapter: str, concept: str,
    language: str, difficulty: int, question_obj: Dict[str, Any], source: str = "llm"
) -> None:
    _require_supabase()

    qid = (question_obj.get("question_id") or "").strip()
    if not qid:
        qid = str(uuid.uuid4())
        question_obj["question_id"] = qid

    payload = {
        "board": board,
        "grade": int(grade),
        "subject": subject,
        "chapter": chapter,
        "concept": concept,
        "chunk_type": "quiz",
        "difficulty": int(difficulty),
        "language": language,
        "chunk_text": json.dumps(question_obj, ensure_ascii=False),
        "tags": ["quiz", (concept or "").lower(), (chapter or "").lower()],
        "source": source,
        "updated_at": utc_now_iso(),
    }

    existing = (
        supabase.table("curriculum_chunks")
        .select("id")
        .eq("board", board)
        .eq("grade", int(grade))
        .eq("subject", subject)
        .eq("chapter", chapter)
        .eq("concept", concept)
        .eq("chunk_type", "quiz")
        .eq("difficulty", int(difficulty))
        .eq("language", language)
        .eq("chunk_text", payload["chunk_text"])
        .limit(1)
        .execute()
    )
    if existing.data:
        return

    supabase.table("curriculum_chunks").insert(payload).execute()

def generate_quiz_via_llm(explain_text: str, difficulty: int, language: str = "en") -> Dict[str, Any]:
    prompt = f"""
Create ONE quiz question from this content.

DIFFICULTY: {difficulty} (1 easiest, 5 hardest)
LANGUAGE: {language}

Return ONLY JSON with keys:
question, options, answer_key, explanation, question_id

Rules:
- options: array of 3 or 4 strings
- answer_key: exactly one of A,B,C,D (must match options length)
- question_id: short string

CONTENT:
{explain_text}
""".strip()

    r = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL_QUIZ", "gpt-4o-mini"),
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )
    txt = (r.choices[0].message.content or "").strip()
    return json.loads(txt)

# ============================================================
# AUTO MASTERY + CONF/STRESS (CLAMPED)
# ============================================================

def estimate_mastery_delta(student_input: str, signals: Dict[str, Any]) -> Tuple[int, str]:
    t = (student_input or "").strip().lower()
    signals = signals or {}

    confused_words = ["confused", "not clear", "can't understand", "cant understand", "i don't get", "dont get", "stuck"]
    success_words = ["i understand", "got it", "now i get it", "makes sense", "clear now", "i got it"]
    quiz_words = ["answer is", "my answer", "option", "a)", "b)", "c)", "d)"]

    emotion = (signals.get("emotion") or "").lower()
    engagement = (signals.get("engagement") or "").lower()

    delta = 0
    reason = "neutral"

    if any(w in t for w in success_words):
        delta += 6
        reason = "self-reported understanding"

    if any(w in t for w in confused_words) or emotion in ["confused", "stuck"]:
        delta -= 4
        reason = "confusion signal"

    if any(w in t for w in quiz_words):
        delta += 1
        if reason == "neutral":
            reason = "attempted answer"

    if engagement == "high":
        delta += 1
    elif engagement == "low":
        delta -= 1

    delta = clamp_delta(delta, 10)
    return delta, reason

def update_mastery(brain: Dict[str, Any], chapter: str, concept: str, delta: int) -> Dict[str, Any]:
    brain = brain or {}
    mastery = brain.get("mastery") or {}
    if not isinstance(mastery, dict):
        mastery = {}

    key = concept_key(chapter, concept) or "Unknown::Unknown"
    cur = mastery.get(key) or {}
    if not isinstance(cur, dict):
        cur = {}

    score = clamp_int(cur.get("score", 0), 0, 100, 0)
    attempts = clamp_int(cur.get("attempts", 0), 0, 10_000, 0)
    streak_correct = clamp_int(cur.get("streak_correct", 0), 0, 10_000, 0)
    streak_wrong = clamp_int(cur.get("streak_wrong", 0), 0, 10_000, 0)

    attempts += 1
    new_score = clamp_int(score + delta, 0, 100, score)

    if delta > 0:
        streak_correct += 1
        streak_wrong = 0
    elif delta < 0:
        streak_wrong += 1
        streak_correct = 0

    mastery[key] = {
        "score": new_score,
        "attempts": attempts,
        "streak_correct": streak_correct,
        "streak_wrong": streak_wrong,
        "last_seen_at": utc_now_iso(),
        "last_delta": int(delta),
    }

    brain["mastery"] = mastery
    return brain

def compute_mastery_rollups(brain: Dict[str, Any], subject: str, chapter: str) -> Dict[str, Any]:
    mastery = brain.get("mastery") or {}
    if not isinstance(mastery, dict) or not mastery:
        brain["mastery_rollup"] = {"subject": subject, "chapter": chapter, "subject_avg": 0, "chapter_avg": 0, "concepts_count": 0}
        return brain

    scores_all: List[int] = []
    scores_chapter: List[int] = []

    for k, v in mastery.items():
        if not isinstance(v, dict):
            continue
        s = v.get("score")
        if not isinstance(s, (int, float)):
            continue
        scores_all.append(int(s))
        if str(k).startswith(f"{chapter}::"):
            scores_chapter.append(int(s))

    subject_avg = int(sum(scores_all) / len(scores_all)) if scores_all else 0
    chapter_avg = int(sum(scores_chapter) / len(scores_chapter)) if scores_chapter else 0

    brain["mastery_rollup"] = {
        "subject": subject,
        "chapter": chapter,
        "subject_avg": subject_avg,
        "chapter_avg": chapter_avg,
        "concepts_count": len(scores_all),
    }
    return brain

# ============================================================
# LEVEL-BASED QUIZ DIFFICULTY (1..5)
# ============================================================

def pick_quiz_difficulty(student_level: int, brain: Dict[str, Any]) -> int:
    lvl = clamp_int(student_level or 2, 1, 5)

    mastery_map = brain.get("mastery") or {}
    avg_mastery = None
    if isinstance(mastery_map, dict) and mastery_map:
        scores = []
        for v in mastery_map.values():
            if isinstance(v, dict) and isinstance(v.get("score"), (int, float)):
                scores.append(int(v["score"]))
        if scores:
            avg_mastery = sum(scores) / len(scores)

    stress = brain.get("stress_score")
    conf = brain.get("confidence_score")
    stress = int(stress) if isinstance(stress, (int, float)) else None
    conf = int(conf) if isinstance(conf, (int, float)) else None

    diff = lvl
    if avg_mastery is not None and avg_mastery < 40:
        diff -= 1
    elif avg_mastery is not None and avg_mastery > 75:
        diff += 1

    if stress is not None and stress >= 70:
        diff -= 1
    if conf is not None and conf >= 70:
        diff += 1

    return clamp_int(diff, 1, 5, lvl)

# ============================================================
# ROOT & HEALTH
# ============================================================

@app.get("/")
def root():
    return {"ok": True, "service": "leaflore-brain", "message": "API running", "time": utc_now_iso()}

@app.get("/health")
def health():
    return {"ok": True, "status": "ok", "time": utc_now_iso()}

# ============================================================
# BRAIN STORAGE ENDPOINTS
# ============================================================

@app.get("/student/{student_id}/brain")
def get_student_brain(student_id: str):
    b = load_brain(student_id)
    return {"student_id": student_id, "brain": b}

@app.post("/student/{student_id}/brain")
def set_student_brain(student_id: str, brain: Dict[str, Any]):
    save_brain(student_id, brain)
    return {"ok": True, "student_id": student_id}

# ============================================================
# GROUP QUIZ ENDPOINTS
# ============================================================

@app.post("/quiz/start")
def quiz_start(req: QuizStartRequest):
    try:
        _require_supabase()
        if not req.student_ids:
            raise HTTPException(status_code=400, detail="student_ids required")

        rep_id = req.representative_student_id or req.student_ids[0]
        rep_brain = load_brain(rep_id)

        stored_level = None
        bd = rep_brain.get("brain_data") or {}
        if isinstance(bd, dict):
            stored_level = bd.get("student_level")

        level = req.student_level or stored_level or 2
        difficulty = pick_quiz_difficulty(int(level), rep_brain)

        row = fetch_quiz_chunk(req.board, req.grade, req.subject, req.chapter, req.concept, req.language, difficulty)
        question_obj = None

        if row and row.get("chunk_text"):
            try:
                question_obj = json.loads(row["chunk_text"])
            except Exception:
                question_obj = {
                    "question": row["chunk_text"],
                    "options": ["A", "B", "C", "D"],
                    "answer_key": "A",
                    "explanation": "Answer stored in curriculum chunk.",
                    "question_id": str(row.get("id") or uuid.uuid4())
                }

        if not question_obj:
            explain_chunks = fetch_explain_chunks(req.board, req.grade, req.subject, req.chapter, req.concept, req.language, limit=4)
            explain_text = "\n".join([c.get("chunk_text", "") for c in explain_chunks if c.get("chunk_text")]) or f"{req.concept} in {req.chapter}"
            question_obj = generate_quiz_via_llm(explain_text, difficulty, req.language)

            try:
                save_quiz_chunk(req.board, req.grade, req.subject, req.chapter, req.concept, req.language, difficulty, question_obj, source="llm")
            except Exception:
                pass

        session_id = str(uuid.uuid4())

        payload = {
            "session_id": session_id,
            "board": req.board,
            "grade": int(req.grade),
            "subject": req.subject,
            "chapter": req.chapter,
            "concept": req.concept,
            "language": req.language,
            "student_ids": req.student_ids,
            "difficulty": int(difficulty),
            "current_question": question_obj,
            "scores": {},
            "updated_at": utc_now_iso(),
        }

        supabase.table("classroom_sessions").insert(payload).execute()

        safe_q = dict(question_obj)
        safe_q.pop("answer_key", None)

        return {"ok": True, "session_id": session_id, "difficulty": difficulty, "question": safe_q}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("quiz_start error")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quiz/{session_id}")
def quiz_get(session_id: str):
    try:
        _require_supabase()
        res = supabase.table("classroom_sessions").select("*").eq("session_id", session_id).limit(1).execute()
        if not res.data:
            raise HTTPException(status_code=404, detail="session not found")
        s = res.data[0] or {}
        q = s.get("current_question") or {}
        if isinstance(q, dict):
            q.pop("answer_key", None)
        return {
            "session_id": session_id,
            "difficulty": s.get("difficulty"),
            "scores": s.get("scores") or {},
            "question": q,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("quiz_get error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quiz/answer")
def quiz_answer(req: QuizAnswerRequest):
    try:
        _require_supabase()

        res = supabase.table("student_brain").upsert(data).execute()

if res.error:
    logger.error("Supabase error: %s", res.error)
    raise Exception(f"Supabase error: {res.error}")

        if not res.data:
            raise HTTPException(status_code=404, detail="session_id not found")

        session = res.data[0] or {}
        q = session.get("current_question") or {}
        answer_key = (q.get("answer_key") or "").strip().upper()
        opts = q.get("options") or []

        user_answer = (req.answer or "").strip()
        user_key = user_answer.upper() if user_answer.upper() in ["A", "B", "C", "D"] else None

        if not user_key and isinstance(opts, list):
            for idx, opt in enumerate(opts[:4]):
                if isinstance(opt, str) and opt.strip().lower() == user_answer.lower():
                    user_key = ["A", "B", "C", "D"][idx]
                    break

        is_correct = bool(user_key and answer_key and user_key == answer_key)

        scores = session.get("scores") or {}
        st = scores.get(req.student_id) or {"correct": 0, "total": 0}
        st["total"] = int(st.get("total", 0)) + 1
        if is_correct:
            st["correct"] = int(st.get("correct", 0)) + 1
        scores[req.student_id] = st

        supabase.table("classroom_sessions").update({"scores": scores, "updated_at": utc_now_iso()}).eq("session_id", req.session_id).execute()

        try:
            supabase.table("quiz_attempts").insert({
                "session_id": req.session_id,
                "student_id": req.student_id,
                "question_id": q.get("question_id"),
                "answer": req.answer,
                "is_correct": is_correct,
            }).execute()
        except Exception:
            pass

        brain = load_brain(req.student_id)
        conf = clamp_int(brain.get("confidence_score", 50), 0, 100, 50)
        stress = clamp_int(brain.get("stress_score", 40), 0, 100, 40)

        if is_correct:
            conf = clamp_int(conf + 5, 0, 100, conf)
            stress = clamp_int(stress - 3, 0, 100, stress)
        else:
            conf = clamp_int(conf - 3, 0, 100, conf)
            stress = clamp_int(stress + 4, 0, 100, stress)

        brain["confidence_score"] = conf
        brain["stress_score"] = stress
        save_brain(req.student_id, brain)

        return {
            "ok": True,
            "is_correct": is_correct,
            "score": scores.get(req.student_id),
            "feedback": ("Correct ✅" if is_correct else "Not correct ❌"),
            "explanation": q.get("explanation", ""),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("quiz_answer error")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# MAIN AI ENDPOINT
# ============================================================

def _llm_call_strict_json(system_prompt: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
    model = os.getenv("OPENAI_MODEL_CHAT", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.6"))

    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_context, ensure_ascii=False)}
        ],
    )
    content = (resp.choices[0].message.content or "").strip()

    try:
        return json.loads(content)
    except Exception:
        repair_prompt = (
            "Your previous output was not valid JSON. "
            "Return ONLY valid JSON with EXACT keys: text, mode, next_action, micro_q, ui_hint, memory_update. "
            "No markdown, no extra keys."
        )
        resp2 = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_context, ensure_ascii=False)},
                {"role": "assistant", "content": content},
                {"role": "user", "content": repair_prompt},
            ],
        )
        content2 = (resp2.choices[0].message.content or "").strip()
        return json.loads(content2)

@app.post("/respond")
def respond(req: BrainRequest):
    try:
        student_id = (req.student_id or "demo").strip()

        # 1) Load brain (request override > DB > empty)
        brain = req.brain or {}
        if (not brain) and supabase is not None:
            brain = load_brain(student_id)

        brain = ensure_turnlock(brain)
        actor = get_actor(req.signals or {})

        # 1.1) Language resolution
        preferred_language = resolve_preferred_language(req.language, req.signals or {}, brain)
        teaching_language = resolve_teaching_language(req.language, req.signals or {}, brain)

        brain["preferred_language"] = preferred_language

        # 2) SOFT TURN-LOCK GATE
        if not is_actor_allowed(brain, actor):
            lock = brain["brain_data"]["telemetry"]["turn_lock"]
            expected = lock.get("expected_actor") or "student"
            turn_id = lock.get("turn_id")

            return {
                "text": (
                    "Thanks for replying. For learning, I want to hear this answer from the student in their own words.\n"
                    "After the student answers, I can share a short parent guidance too."
                ),
                "mode": "parent_support" if actor == "parent" else "support",
                "next_action": "wait_for_student",
                "micro_q": "Can the student answer in one line: what does this concept do?",
                "ui_hint": "headline=Waiting for Student | sub=Please let the student answer | meta=Leaflore Live Class | badge1=Conf 50 | badge2=Mast 0 | badge3=Stress 40",
                "memory_update": {
                    "brain_data": {
                        "telemetry": {
                            "last_blocked_actor": actor,
                            "last_blocked_turn_id": turn_id,
                            "expected_actor": expected
                        }
                    },
                    "confidence_score": clamp_int(brain.get("confidence_score", 50), 0, 100, 50),
                    "stress_score": clamp_int(brain.get("stress_score", 40), 0, 100, 40),
                }
            }

        brain = clear_turn_lock(brain)

        # 3) Fetch chunks (teaching_language)
        chunks = []
        if supabase is not None:
            chunks = fetch_explain_chunks(
                board=req.board or "ICSE",
                grade=int(req.grade or 6),
                subject=req.subject or "Science",
                chapter=req.chapter or "",
                concept=req.concept or "",
                language=teaching_language,
                limit=6
            )

        # 4) Build context
        user_context = {
            "board": req.board,
            "grade": req.grade,
            "subject": req.subject,
            "chapter": req.chapter,
            "concept": req.concept,
            "preferred_language": preferred_language,
            "teaching_language": teaching_language,
            "language": teaching_language,
            "student_input": req.student_input,
            "signals": req.signals or {},
            "brain": brain,
            "chunks": chunks
        }

        parsed = _llm_call_strict_json(SYSTEM_PROMPT, user_context)

        required_keys = {"text", "mode", "next_action", "micro_q", "ui_hint", "memory_update"}
        if not required_keys.issubset(parsed.keys()):
            raise ValueError(f"Missing required keys in AI response. Got keys={list(parsed.keys())}")

        # 5) Merge memory_update into brain
        memory_update = parsed.get("memory_update") or {}
        new_brain = deep_merge(brain, memory_update)
        new_brain["preferred_language"] = preferred_language

        # 6) Auto mastery + conf/stress clamp
        delta, reason = estimate_mastery_delta(req.student_input, req.signals or {})
        new_brain = update_mastery(new_brain, req.chapter or "", req.concept or "", delta)

        conf = clamp_int(new_brain.get("confidence_score", 50), 0, 100, 50)
        stress = clamp_int(new_brain.get("stress_score", 40), 0, 100, 40)

        conf = clamp_int(conf + delta, 0, 100, conf)
        stress = clamp_int(stress - delta, 0, 100, stress)

        new_brain["confidence_score"] = conf
        new_brain["stress_score"] = stress

        new_brain = compute_mastery_rollups(new_brain, req.subject or "", req.chapter or "")

        # 7) Telemetry + lock next turn
        new_brain = ensure_turnlock(new_brain)
        bd = new_brain.get("brain_data") or {}
        if not isinstance(bd, dict):
            bd = {}
        bd.setdefault("version", "prod-1")
        bd["last_interaction_at"] = utc_now_iso()

        tel = bd.get("telemetry") or {}
        if not isinstance(tel, dict):
            tel = {}

        tel["last_ai"] = {
            "mode": parsed.get("mode"),
            "next_action": parsed.get("next_action"),
            "ui_hint": parsed.get("ui_hint"),
            "micro_q": parsed.get("micro_q"),
            "preferred_language": preferred_language,
            "teaching_language": teaching_language,
        }
        tel["last_mastery_reason"] = reason
        bd["telemetry"] = tel
        new_brain["brain_data"] = bd

        if actor == "parent":
            new_brain = set_turn_lock(new_brain, expected_actor="parent", purpose="parent_chat")
        else:
            new_brain = set_turn_lock(new_brain, expected_actor="student", purpose="micro_q")

        if supabase is not None:
            save_brain(student_id, new_brain)

        return parsed

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("respond error")
        raise HTTPException(status_code=500, detail=str(e))
