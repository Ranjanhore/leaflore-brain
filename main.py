import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ===== Structured Request =====
class RespondRequest(BaseModel):
    student_id: str
    board: str
    grade: str
    subject: str
    chapter: str
    concept: str
    student_input: str
    signals: Optional[Dict] = None
    class_phase: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/respond")
def respond(req: RespondRequest):

    if not req.student_input:
        raise HTTPException(status_code=400, detail="student_input required")

    # Extract signals safely
    emotion = req.signals.get("emotion") if req.signals else "neutral"
    engagement = req.signals.get("engagement") if req.signals else "normal"

    # Dynamic system prompt
    system_prompt = f"""
You are Leaflore AI Teacher.

Student Details:
- Board: {req.board}
- Grade: {req.grade}
- Subject: {req.subject}
- Chapter: {req.chapter}
- Concept: {req.concept}
- Emotion: {emotion}
- Engagement Level: {engagement}

Instructions:
- Explain slowly like a story.
- Adjust difficulty to grade {req.grade}.
- If emotion is confused, simplify.
- If engagement is low, make it more interactive.
- Ask one short question at the end.
"""

    r = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.student_input},
        ],
        max_output_tokens=400,
    )

    out = r.output_text if hasattr(r, "output_text") else ""

    return {"text": out}
