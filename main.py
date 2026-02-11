import os, json
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Leaflore NeuroTutor Brain")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class RespondReq(BaseModel):
    user_id: str
    chapter: str
    concept: str
    student_input: str
    recent: Optional[Dict[str, Any]] = None
    signals: Optional[Dict[str, Any]] = None
    lang: str = "en"

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/respond")
def respond(req: RespondReq):
    try:
        system = """
You are Leaflore NeuroTutor: a science tutor + learning psychology coach.
You do NOT diagnose or provide therapy.
You adapt tone based on learning signals.
Return ONLY strict JSON with keys:
text, mode, next_action, micro_q, ui_hint, memory_update
"""
        user = {
            "chapter": req.chapter,
            "concept": req.concept,
            "student_input": req.student_input,
            "recent": req.recent or {},
            "signals": req.signals or {},
            "lang": req.lang,
        }

        completion = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Context:\n{json.dumps(user)}\n\nOutput JSON only."},
            ],
            response_format={"type": "json_object"},
        )

        return json.loads(completion.choices[0].message.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"agent_failed: {e}")
