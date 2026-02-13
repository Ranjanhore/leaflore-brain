import os
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from supabase import create_client, Client

# ============================================================
# APP
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
    raise RuntimeError("Missing OPENAI_API_KEY env var")

client = OpenAI(api_key=OPENAI_API_KEY)

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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
    student_input: str

    # Signals:
    # - actor: student/parent/other/unknown
    # - emotion, engagement, voice_id, reply_to_turn_id, etc.
    # - language / preferred_language
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
# SYSTEM MASTER PROMPT (TEACHER BRAIN)
# ============================================================

SYSTEM_PROMPT = """
You are Leaflore NeuroMentor — an elite, warm, calm, highly intelligent teacher, learning scientist, career mentor, guardian guide, and emotional support coach.

You are NOT a medical doctor.
You DO NOT diagnose, treat, label, or provide therapy.
No psychiatric claims.
All stress/confidence references are NON-CLINICAL learning indicators.

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
CURRICULUM CHUNKS (MANDATORY)
====================================================
You may receive "chunks" (curriculum content).
Use chunks as primary source of truth when available.
If chunks are missing, teach generally but keep it simple and ask ONE micro-question.
Prefer:
1) explain
2) misconception
3) example
4) quiz

====================================================
ACTOR ROUTING
====================================================
You will receive signals.actor.

If signals.actor == "parent":
- Use respectful, polite tone.
- Give parent-friendly guidance (routines, communication tips, gentle motivation).
- Do NOT test the parent like a student.
- micro_q should be a parent question (routine/observation/goal).
- Keep it practical, short, and supportive.
- Still follow safety boundaries (no diagnosis/therapy/medical claims).

If signals.actor == "student":
- Teach the concept clearly.
- Ask the student micro-question (one).
- You may be firm/bold if needed but never rude or shaming.

If signals.actor == "other" or "unknown":
- Ask a gentle clarification in the micro_q.

====================================================
SAFETY BOUNDARIES
====================================================
- No diagnosis.
- No therapy language.
- No medical claims.
- No psychiatric labels.
- If self-harm mentioned → calmly encourage contacting a trusted adult.


====================================================
AUTO SCREEN HEADLINE GENERATION LOGIC (v2)
====================================================

UI CONTROL:
You must use the existing JSON key "ui_hint" to carry a compact UI payload as a STRING.
Format:
ui_hint = "headline=<...> | sub=<...> | meta=<...> | badge1=<...> | badge2=<...> | badge3=<...>"

Do NOT add new JSON keys.
Do NOT output markdown.

GENERATE EVERY TURN:
Generate headline/sub/meta/badges EVERY TURN.

CHAR LIMITS:
- headline: max 52 chars
- sub: max 72 chars
- meta: max 72 chars
- each badge: max 18 chars

DATA SOURCES:
- Student name:
  - signals.student_name OR brain.student_name
- End-of-class flag:
  - signals.class_end == true OR signals.event == "class_end"
  - OR next_action == "end_class"
- Score fields (if present):
  - brain.mastery_rollup.chapter_avg
  - brain.mastery_rollup.subject_avg
  - brain.confidence_score
  - brain.stress_score
  - brain.mastery[<chapter::concept>].score

DEFAULTS IF MISSING:
confidence=50
stress=40
mastery=0

MASTERY PICK PRIORITY:
1) mastery score for key "<chapter>::<concept>" if exists
2) brain.mastery_rollup.chapter_avg if exists
3) brain.mastery_rollup.subject_avg if exists
4) 0

HEADLINE RULES (priority order):
A) If end-of-class:
   - If student_name known: "Today’s Score, <Name>"
   - Else: "Today’s Score"
B) Else if next_action == "ask_name":
   - If student_name known: "Welcome, <Name>!"
   - Else: "Welcome to Leaflore"
C) Else if next_action == "ask_language":
   - If student_name known: "<Name>, Choose Language"
   - Else: "Choose Your Language"
D) Else if mode == "quiz":
   - If student_name known: "Quick Quiz, <Name>!"
   - Else: "Quick Quiz Time"
E) Else if mode == "parent_support":
   - "Parent Guidance"
F) Else if ui_hint indicates gate/wait:
   - "Waiting for Student"
G) Else if mode == "teach":
   - If chapter present: "Chapter: <chapter>"
   - Else: "Let’s Learn Together"
H) Else:
   - If student_name known: "Let’s go, <Name>!"
   - Else: "Let’s Learn Together"

SUBTITLE (sub) RULES:
A) If end-of-class:
   - "Confidence <conf> • Mastery <mast> • Stress <stress>"
B) Else if asking name:
   - "Tell me your name to start the class"
C) Else if asking language:
   - "English, Hindi, or Both (Hinglish)"
D) Else if signals.emotion indicates confused/stuck:
   - "No stress — we’ll go step by step"
E) Else if signals.emotion indicates confident:
   - "Great! Let’s level up a little"
F) Else if mode == "teach":
   - If concept present: "Today: <concept> in simple steps"
   - Else: "Today’s lesson in small steps"
G) Else:
   - "Small chunks • Quick questions • Big progress"

META RULE:
meta should show class info using only available fields:
Format examples:
- "ICSE • Grade 6 • Science • Ch 1"
- "ICSE • Grade 6 • Science"
If chapter number unknown, omit it.

BADGES (text-only):
badge1: "Conf <0-100>"
badge2: "Mast <0-100>"
badge3: "Stress <0-100>"

Use computed values with clamping 0..100.

FINAL UI STRING:
ui_hint must be exactly:
"headline=... | sub=... | meta=... | badge1=... | badge2=... | badge3=..."

No extra keys.
No extra separators besides " | ".


====================================================
====================================================
MEMORY RULE
====================================================
Always update memory_update safely.
Confidence and stress change max ±10 per turn (non-clinical).
""".strip()

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

    # common variants
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
    """
    Stores the user's preference as-is (en/hi/hinglish).
    Priority:
    1) signals.preferred_language OR signals.language
    2) request.language
    3) brain.preferred_language
    Default: 'hinglish'
    """
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
    """
    Teaching language rule:
    - If preferred is Hindi (hi) => teach in Hinglish
    Otherwise teach in the chosen language.
    """
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
    """Deep merge dictionaries: b overlays a."""
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
# TURN-LOCK (SOFT GATE) HELPERS
# ============================================================

def ensure_turnlock(brain: Dict[str, Any]) -> Dict[str, Any]:
    brain = brain or {}
    bd = brain.get("brain_data") or {}
    tel = bd.get("telemetry") or {}
    lock = tel.get("turn_lock") or {}

    if not isinstance(lock, dict):
        lock = {}

    lock.setdefault("enabled", False)
    lock.setdefault("expected_actor", None)   # "student" / "parent" / "other"
    lock.setdefault("turn_id", None)
    lock.setdefault("purpose", None)          # "micro_q" / "quiz" / "parent_chat"
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
# CURRICULUM CHUNKS (EXPLAIN/QUIZ) + AUTOSAVE
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
        model="gpt-4o-mini",
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

    return clamp_int(diff, 1, 5)

# ============================================================
# ROOT & HEALTH
# ============================================================

@app.get("/")
def root():
    return {"ok": True, "service": "leaflore-brain", "message": "API is running. Use /docs or POST /respond"}

@app.get("/health")
def health():
    return {"ok": True}

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
        try:
            bd = rep_brain.get("brain_data") or {}
            if isinstance(bd, dict):
                stored_level = bd.get("student_level")
        except Exception:
            stored_level = None

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
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quiz/answer")
def quiz_answer(req: QuizAnswerRequest):
    try:
        _require_supabase()

        res = supabase.table("classroom_sessions").select("*").eq("session_id", req.session_id).limit(1).execute()
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

        # optional attempt log
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

        # update student brain lightly
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
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# MAIN AI ENDPOINT
# ============================================================

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

        # 1.1) Language resolution (store preference, teach in resolved language)
        preferred_language = resolve_preferred_language(req.language, req.signals or {}, brain)
        teaching_language = resolve_teaching_language(req.language, req.signals or {}, brain)

        # Store preference in brain (so it persists to Supabase column preferred_language)
        brain["preferred_language"] = preferred_language

        # 2) SOFT TURN-LOCK GATE
        if not is_actor_allowed(brain, actor):
            lock = brain["brain_data"]["telemetry"]["turn_lock"]
            expected = lock.get("expected_actor") or "student"
            turn_id = lock.get("turn_id")

            return {
                "text": (
                    "Thanks for replying. For learning, I want to hear this answer from the student in their own words. "
                    "After the student answers, I can share a short parent guidance too."
                ),
                "mode": "parent_support" if actor == "parent" else "support",
                "next_action": "wait_for_student",
                "micro_q": "Can the student answer in one line: what does this concept do?",
                "ui_hint": "gate",
                "memory_update": {
                    "brain_data": {
                        "telemetry": {
                            "last_blocked_actor": actor,
                            "last_blocked_turn_id": turn_id,
                            "expected_actor": expected
                        }
                    }
                }
            }

        # allowed actor answered -> clear lock
        brain = clear_turn_lock(brain)

        # 3) Fetch curriculum explain chunks (USE teaching_language)
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

        # 4) Build LLM input context
        user_context = {
            "board": req.board,
            "grade": req.grade,
            "subject": req.subject,
            "chapter": req.chapter,
            "concept": req.concept,

            # include both
            "preferred_language": preferred_language,   # what user selected/stored
            "teaching_language": teaching_language,     # what teacher will speak
            "language": teaching_language,              # keep compatibility

            "student_input": req.student_input,
            "signals": req.signals or {},
            "brain": brain,
            "chunks": chunks
        }

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.6,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_context, ensure_ascii=False)}
            ],
        )

        content = (response.choices[0].message.content or "").strip()

        # 5) Strict JSON parse + required keys
        parsed = json.loads(content)
        required_keys = {"text", "mode", "next_action", "micro_q", "ui_hint", "memory_update"}
        if not required_keys.issubset(parsed.keys()):
            raise ValueError(f"Missing required keys in AI response. Got keys={list(parsed.keys())}")

        # 6) Merge memory_update into brain (deep)
        memory_update = parsed.get("memory_update") or {}
        new_brain = deep_merge(brain, memory_update)

        # Ensure preferred_language persists even if memory_update overwrote brain
        new_brain["preferred_language"] = preferred_language

        # 7) Auto mastery + conf/stress clamp
        delta, reason = estimate_mastery_delta(req.student_input, req.signals or {})
        new_brain = update_mastery(new_brain, req.chapter or "", req.concept or "", delta)

        conf = clamp_int(new_brain.get("confidence_score", 50), 0, 100, 50)
        stress = clamp_int(new_brain.get("stress_score", 40), 0, 100, 40)

        conf = clamp_int(conf + delta, 0, 100, conf)
        stress = clamp_int(stress - delta, 0, 100, stress)

        new_brain["confidence_score"] = conf
        new_brain["stress_score"] = stress

        # 8) Mastery rollups (Subject + Chapter)
        new_brain = compute_mastery_rollups(new_brain, req.subject or "", req.chapter or "")

        # 9) Telemetry + set lock for next turn
        new_brain = ensure_turnlock(new_brain)
        bd = new_brain.get("brain_data") or {}
        if not isinstance(bd, dict):
            bd = {}
        bd.setdefault("version", "v2")
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

        # Soft lock default:
        # - parent => lock parent chat
        # - student/unknown/other => lock student micro_q
        if actor == "parent":
            new_brain = set_turn_lock(new_brain, expected_actor="parent", purpose="parent_chat")
        else:
            new_brain = set_turn_lock(new_brain, expected_actor="student", purpose="micro_q")

        # 10) Persist brain
        if supabase is not None:
            save_brain(student_id, new_brain)

        return parsed

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))