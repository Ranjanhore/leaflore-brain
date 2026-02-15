from __future__ import annotations

from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --------------------------------
# App
# --------------------------------

app = FastAPI(title="Leaflore Brain API")

# --------------------------------
# CORS (Fixes Lovable 403 + preflight)
# --------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow Lovable preview + publish
    allow_credentials=False,      # must be False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------
# Models
# --------------------------------

class RespondRequest(BaseModel):
    action: str = Field(default="respond")
    student_input: str
    student_id: Optional[str] = None
    board: Optional[str] = None


# --------------------------------
# Routes
# --------------------------------

@app.get("/")
def root():
    return {
        "service": "Leaflore Brain API",
        "health": "/health",
        "respond": "/respond",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/respond")
def respond(body: RespondRequest):
    # Simple safe reply for now
    reply_text = f"Teacher says: I heard you say '{body.student_input}'."

    return {
        "text": reply_text,
        "status": "success"
    }
