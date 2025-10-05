# suppress_api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
import time
import threading
import os

app = FastAPI(title="Suppress Control API", version="1.0")

# Open CORS so your other laptop / mobile app can call this over the hotspot

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or your dev IPs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SuppressRequest(BaseModel):
    mode: Literal["drop", "keep"] = Field(default="drop", description="drop=remove targets; keep=isolate targets")
    classes: List[str] = Field(default_factory=list, description="Class/label names to drop/keep")
    profile: Optional[str] = Field(default=None, description="Optional profile name (e.g., 'Focus', 'Work Call')")

class SuppressResponse(BaseModel):
    ok: bool
    mode: Literal["drop", "keep"]
    classes: List[str]
    profile: Optional[str]
    version: int
    updated_at: float

# ---- shared state the DSP/model can read ----
class _State:
    def __init__(self):
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {
            "mode": "drop",
            "classes": [],
            "profile": None,
            "version": 0,
            "updated_at": time.time(),
        }

    def set(self, *, mode: str, classes: List[str], profile: Optional[str] = None):
        with self._lock:
            self._data["mode"] = mode
            self._data["classes"] = [c.strip() for c in classes]
            self._data["profile"] = profile
            self._data["version"] += 1
            self._data["updated_at"] = time.time()
            return dict(self._data)

    def clear(self):
        return self.set(mode="drop", classes=[], profile=None)

    def get(self):
        with self._lock:
            return dict(self._data)

STATE = _State()

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/suppress/current", response_model=SuppressResponse)
def get_current():
    d = STATE.get()
    return SuppressResponse(ok=True, mode=d["mode"], classes=d["classes"],
                            profile=d["profile"], version=d["version"], updated_at=d["updated_at"])

@app.post("/suppress/set", response_model=SuppressResponse)
def set_suppress(req: SuppressRequest):
    if req.mode not in ("drop", "keep"):
        raise HTTPException(400, "mode must be 'drop' or 'keep'")
    d = STATE.set(mode=req.mode, classes=req.classes, profile=req.profile)
    return SuppressResponse(ok=True, mode=d["mode"], classes=d["classes"],
                            profile=d["profile"], version=d["version"], updated_at=d["updated_at"])

@app.post("/suppress/clear", response_model=SuppressResponse)
def clear_suppress():
    d = STATE.clear()
    return SuppressResponse(ok=True, mode=d["mode"], classes=d["classes"],
                            profile=d["profile"], version=d["version"], updated_at=d["updated_at"])

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("SUPPRESS_API_HOST", "0.0.0.0")
    port = int(os.getenv("SUPPRESS_API_PORT", "8000"))
    log_level = os.getenv("SUPPRESS_API_LOG_LEVEL", "info")
    uvicorn.run(app, host=host, port=port, log_level=log_level)
