#!/usr/bin/env python3
"""
FastAPI WebSocket server for real-time YAMNet audio classification
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
import json
import numpy as np
import os
from collections import deque
from math import gcd
import tensorflow as tf
import tensorflow_hub as hub
from scipy.signal import resample_poly

# Configure logging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# YAMNet Configuration
TARGET_SR = 16000
WIN_SECS = 0.975
HOP_SECS = 0.10
FRAME_LEN = int(WIN_SECS * TARGET_SR)

print("Loading YAMNet…")
yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

def load_labels():
    try:
        p = yamnet.class_map_path().numpy().decode("utf-8")
        with tf.io.gfile.GFile(p, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return [ln.split(",")[-1].strip('"') for ln in lines[1:]]
    except Exception:
        return [f"class_{i}" for i in range(521)]

LABELS = load_labels()

class EMA:
    def __init__(self, alpha=0.6):
        self.alpha = float(alpha)
        self.v = None
    def update(self, x: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha * x + (1 - self.alpha) * self.v
        return self.v

# Global variables for real-time processing
ring = deque(maxlen=FRAME_LEN)
level_ema = 0.0
ema = EMA(alpha=0.6)

def meter_update(x: np.ndarray):
    global level_ema
    rms = np.sqrt(np.mean(np.square(x))) + 1e-12
    level_ema = 0.9 * level_ema + 0.1 * rms

print("✅ YAMNet model loaded successfully")
print(f"📊 Loaded {len(LABELS)} audio classes")

# Warm up model
_ = yamnet(np.zeros(FRAME_LEN, dtype=np.float32))
print("🔥 Model warmed up")

app = FastAPI(title="Real-time YAMNet WebSocket API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active connections
active_connections: list[WebSocket] = []

def classify_audio_realtime(audio_float: np.ndarray) -> dict:
    """Classify audio using real-time YAMNet"""
    try:
        # Update level meter
        meter_update(audio_float)
        
        # Fill rolling window
        ring.extend(audio_float)
        
        if len(ring) < FRAME_LEN:
            return {"error": "Not enough samples", "samples": len(ring)}
        
        # Run YAMNet
        window = np.array(ring, dtype=np.float32)
        scores, _, _ = yamnet(window)
        avg = scores.numpy().mean(axis=0)
        if ema:
            avg = ema.update(avg)
        
        # Get predictions
        min_prob = 0.12
        items = []
        for i, p in enumerate(avg):
            name = LABELS[i]
            if p < min_prob:
                continue
            items.append((name, float(p)))
        
        if items:
            items.sort(key=lambda x: x[1], reverse=True)
            items = items[:5]
        else:
            i_top = int(np.argmax(avg))
            items = [(LABELS[i_top], float(avg[i_top]))]
        
        # Print to terminal
        lvl_db = 20*np.log10(level_ema + 1e-6)
        line = " | ".join(f"{n}: {p:.2f}" for n, p in items)
        print(f"[level ~ {lvl_db:5.1f} dB] {line}")
        
        return {
            "top_class": items[0][0],
            "top_confidence": items[0][1],
            "level_db": lvl_db,
            "predictions": [{"class": name, "confidence": prob} for name, prob in items]
        }
        
    except Exception as e:
        return {"error": f"Classification failed: {str(e)}"}

@app.get("/")
async def root():
    return {"message": "Real-time YAMNet WebSocket API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "connections": len(active_connections)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    print(f"✅ Client connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            # Receive audio data from client
            data = await websocket.receive_text()
            
            try:
                # Decode Base64 and convert to float32
                audio_bytes = base64.b64decode(data)
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32767.0
                
                # Classify the audio chunk
                result = classify_audio_realtime(audio_float32)
                
                # Send result back to client
                await websocket.send_text(json.dumps({
                    "type": "classification",
                    "result": result
                }))
                
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                }))
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        print(f"❌ Client disconnected. Total connections: {len(active_connections)}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

if __name__ == "__main__":
    print("🎤 Starting YAMNet WebSocket Server...")
    print("📍 WebSocket: ws://172.20.10.2:8000/ws")
    print("🔍 Health: http://172.20.10.2:8000/health")
    print("📚 Docs: http://172.20.10.2:8000/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
