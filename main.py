import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Leaflore Brain API")

# -----------------------------
# CORS (ALLOW LOVABLE + LOCAL)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def root():
    return {
        "service": "Leaflore Brain API",
        "health": "/health",
        "respond": "/respond"
    }

# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Request Model
# -----------------------------
class TextRequest(BaseModel):
    text: str

# -----------------------------
# Respond Endpoint (POST ONLY)
# -----------------------------
@app.post("/respond")
def respond(request: TextRequest):
    user_text = request.text

    # Simple demo teacher logic
    reply = f"Hello! You said: '{user_text}'. Let's learn together."

    return {
        "teacher_reply": reply,
        "status": "success"
    }

# -----------------------------
# Render PORT Binding
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
