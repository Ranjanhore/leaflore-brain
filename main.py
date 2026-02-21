import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Req(BaseModel):
    text: str
    # optional context fields you can add later:
    chapter_id: str | None = None
    chunk_id: str | None = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/respond")
def respond(req: Req):
    if not req.text:
        raise HTTPException(status_code=400, detail="text is required")

    r = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "You are Leaflore teacher. Explain slowly like a story. Ask 1 short question at the end."
            },
            {"role": "user", "content": req.text},
        ],
    )
    # safest extraction
    out = r.output_text if hasattr(r, "output_text") else ""
    return {"text": out}
