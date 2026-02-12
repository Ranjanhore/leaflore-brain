import os
import json
from typing import Optional, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

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
# OPENAI CLIENT
# ============================================================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
# REQUEST MODEL
# ============================================================

class BrainRequest(BaseModel):
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
# MAIN AI ENDPOINT
# ============================================================

@app.post("/respond")
def respond(req: BrainRequest):

    try:
        user_context = f"""
Chapter: {req.chapter}
Concept: {req.concept}
Student Input: {req.student_input}
Signals: {json.dumps(req.signals)}
Brain Memory: {json.dumps(req.brain)}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.6,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_context}
            ]
        )

        content = response.choices[0].message.content.strip()

        # Enforce strict JSON output
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

        return parsed

    except Exception as e:
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
