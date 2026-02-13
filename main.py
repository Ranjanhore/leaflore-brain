
import os
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# IMPORTANT: app must exist at module import time
app = FastAPI(title="Leaflore Brain API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"status": "brain alive"}