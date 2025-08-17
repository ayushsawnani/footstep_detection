# ==============================
# File: server.py (FastAPI backend)
# Purpose: Receive live audio chunks (~0.5s), buffer per session, run the
#          last 1.5s window through the CNN, and return footstep probability.
# ==============================

import os
import io
import json
from typing import Dict

import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

APP_SAMPLE_RATE_DEFAULT = 48000  # typical browser sample rate
TARGET_SR = 16000
WIN_S = 1.5
N_FFT = 1024
HOP_LEN = 256
FMAX = 200.0

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev; tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load model + norm on startup ----
MODEL_PATH = os.environ.get("MODEL_PATH", "models/final_cnn.keras")
NORM_PATH = os.environ.get("NORM_PATH", "models/norm.json")

_model = tf.keras.models.load_model(MODEL_PATH)  # type: ignore
with open(NORM_PATH) as f:
    _norm = json.load(f)
MEAN = float(_norm["mean"]) if "mean" in _norm else 0.0
STD = float(_norm["std"]) if "std" in _norm else 1.0

# Per-session rolling buffers (float32 @ incoming sr)
session_buffers: Dict[str, np.ndarray] = {}
session_rates: Dict[str, int] = {}


def preprocess_window(y: np.ndarray, sr_in: int) -> np.ndarray:
    """Take raw mono PCM, resample->TARGET_SR, compute <=200 Hz log-spec window."""
    if sr_in != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr_in, target_sr=TARGET_SR)
    # STFT power -> dB, focus band
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LEN, window="hann")) ** 2
    freqs = librosa.fft_frequencies(sr=TARGET_SR, n_fft=N_FFT)
    idx = np.where(freqs <= FMAX)[0]
    S_focus = S[idx, :]
    S_db = librosa.power_to_db(S_focus, ref=np.max)
    X = (S_db - MEAN) / (STD + 1e-6)
    # add channels
    return X.astype(np.float32)


@app.post("/predict_chunk")
async def predict_chunk(request: Request):
    session_id = request.headers.get("X-Session", "default")
    try:
        sr = int(request.headers.get("X-SampleRate", APP_SAMPLE_RATE_DEFAULT))
    except Exception:
        sr = APP_SAMPLE_RATE_DEFAULT
    try:
        t_end_ms = int(request.headers.get("X-ElapsedMs", "0"))
    except Exception:
        t_end_ms = 0

    raw = await request.body()
    if len(raw) % 4 != 0:
        # Expect Float32 little-endian
        return JSONResponse({"error": "Invalid buffer length"}, status_code=400)

    chunk = np.frombuffer(raw, dtype=np.float32)

    # Append to rolling buffer
    buf = session_buffers.get(session_id)
    if buf is None:
        buf = chunk
    else:
        buf = np.concatenate([buf, chunk])
    session_buffers[session_id] = buf
    session_rates[session_id] = sr

    # Build a 1.5s window ending now
    win_len = int(WIN_S * sr)
    if buf.shape[0] >= win_len:
        window = buf[-win_len:]
    else:
        pad = win_len - buf.shape[0]
        window = np.pad(buf, (pad, 0))

    # Preprocess -> predict
    X = preprocess_window(window, sr_in=sr)
    X_in = np.expand_dims(X, axis=(0, -1))
    prob = float(_model.predict(X_in, verbose=0)[0, 0])

    return {"prob": prob, "t_end_ms": t_end_ms}


@app.post("/reset_session")
async def reset_session(request: Request):
    session_id = request.headers.get("X-Session", "default")
    session_buffers.pop(session_id, None)
    session_rates.pop(session_id, None)
    return {"ok": True}
