from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="Leaflore Brain API", version="1.0.0")

# -----------------------------------------------------------------------------
# CORS (Lovable preview/publish + local dev + allow-all safe default)
# IMPORTANT:
# - If allow_origins=["*"], allow_credentials MUST be False.
# - This setup accepts Lovable preview/publish domains + localhost + anything else.
# -----------------------------------------------------------------------------
LOVABLE_ORIGIN_REGEX = r"^https://([a-z0-9-]+\.)?lovable(app|project)\.com$"
LOCAL_ORIGIN_REGEX = r"^https?://localhost(:\d+)?$|^https?://127\.0\.0\.1(:\d+)?$"

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=f"({LOVABLE_ORIGIN_REGEX})|({LOCAL_ORIGIN_REGEX})|^https://.*$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class Signals(BaseModel):
    actor: Optional[str] = "student"
    emotion: Optional[str] = None
    engagement: Optional[str] = None


class RespondRequest(BaseModel):
    action: str = Field(default="respond")
    student_input: str

    # Optional context (your frontend already sends many of these sometimes)
    student_id: Optional[str] = None
    board: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    chapter: Optional[str] = None
    concept: Optional[str] = None
    language: Optional[str] = "english"
    signals: Optional[Signals] = None

    # Optional "memory" payloads if you add later
    memory: Optional[Dict[str, Any]] = None


class RespondResponse(BaseModel):
    text: str
    mode: str = "teach"
    meta: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _teacher_system_prompt() -> str:
    return """
You are Leaflore Teacher — a warm, real, human-sounding teacher and mentor for kids.

Style & tone:
- Sound like a kind human chatting naturally (not robotic, not template-y).
- No “I heard you say …”, no repeating the student's sentence unless needed for clarity.
- Be humble, friendly, calm, and encouraging.
- Keep replies concise (2–6 short sentences) unless the student asks for detail.
- Ask exactly ONE gentle follow-up question when helpful.

Universal teaching:
- You can help with ANY subject (science, math, English, history, art, life skills).
- If the student asks a casual personal question (e.g., “did you eat lunch?”), answer briefly like a human,
  then softly guide back: “Want to learn something or ask me anything?”
- If context is provided (grade/subject/chapter), adapt examples and difficulty to that level.
- If student is confused, break into small steps and offer a simple example.

“Neuro-analytics + balanced teacher” behavior:
- Notice emotions (confused, anxious, excited) from the text and respond supportively.
- Use gentle encouragement, grounding phrases, and clear structure.
- Do NOT claim to be a doctor/therapist. You may use supportive coaching language.
- If the student mentions self-harm, abuse, or immediate danger: advise talking to a trusted adult immediately.

Output:
- Return ONLY the final teacher reply text. No JSON. No headings. No bullet spam.
""".strip()


async def _openai_reply(req: RespondRequest) -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        # Fallback (still “human”)
        msg = _clean(req.student_input)
        if not msg:
            return "Tell me your question, and I’ll help."
        # Casual question handling
        if any(k in msg.lower() for k in ["lunch", "dinner", "breakfast", "how are you", "what's up", "whats up"]):
            return "I’m doing good 🙂 I had something simple. What would you like to learn or ask today?"
        return "Got it. Tell me which subject this is for, and what exactly is confusing you?"

    # Build context line (short)
    ctx_bits = []
    if req.grade:
        ctx_bits.append(f"Grade {req.grade}")
    if req.board:
        ctx_bits.append(str(req.board))
    if req.subject:
        ctx_bits.append(str(req.subject))
    if req.chapter:
        ctx_bits.append(f"Chapter: {req.chapter}")
    if req.concept:
        ctx_bits.append(f"Concept: {req.concept}")
    ctx = " | ".join([b for b in ctx_bits if b])

    # Detect very light emotion hints (optional)
    emotion_hint = ""
    s = (req.student_input or "").lower()
    if any(w in s for w in ["i'm scared", "i am scared", "worried", "anxious", "nervous", "panic"]):
        emotion_hint = "Student seems anxious. Be extra reassuring and gentle."
    elif any(w in s for w in ["confused", "i don't get", "dont get", "hard", "difficult"]):
        emotion_hint = "Student seems confused. Use tiny steps and a simple example."
    elif any(w in s for w in ["yay", "awesome", "excited", "great"]):
        emotion_hint = "Student seems excited. Match the energy while staying clear."

    user_msg = req.student_input.strip()
    if ctx:
        user_msg = f"Context: {ctx}\n\nStudent: {req.student_input.strip()}"
    if emotion_hint:
        user_msg = f"{emotion_hint}\n\n{user_msg}"

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _teacher_system_prompt()},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.7,
        "max_tokens": 220,
    }

    timeout = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )

    if r.status_code >= 400:
        # surface something readable for debugging
        try:
            err = r.json()
        except Exception:
            err = {"error": r.text}
        raise HTTPException(status_code=502, detail={"upstream": "openai", "error": err})

    data = r.json()
    text = data["choices"][0]["message"]["content"]
    return (text or "").strip()


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"service": "Leaflore Brain API", "health": "/health", "respond": "/respond"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/respond", response_model=RespondResponse)
async def respond(req: RespondRequest):
    if _clean(req.action).lower() != "respond":
        raise HTTPException(status_code=400, detail="Invalid action. Use action='respond'.")

    if not _clean(req.student_input):
        raise HTTPException(status_code=400, detail="student_input is required.")

    reply_text = await _openai_reply(req)

    # Final safety cleanup: avoid the exact unwanted prefix if model ever does it
    bad_prefixes = [
        "hi! i heard you say",
        "i heard you say",
        "you said:",
    ]
    low = reply_text.lower().strip()
    for bp in bad_prefixes:
        if low.startswith(bp):
            # remove first line / prefix-like opener
            reply_text = re.sub(r"^(hi!?\s*)?(i heard you say|you said:)\s*['\"]?.*?['\"]?\s*[,:\-–]*\s*",
                                "", reply_text, flags=re.IGNORECASE).strip()
            break

    return RespondResponse(
        text=reply_text,
        mode="teach",
        meta={
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "has_openai_key": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        },
    )
