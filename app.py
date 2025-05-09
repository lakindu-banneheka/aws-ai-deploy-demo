import io
import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from mangum import Mangum

# --- initialize ---
app = FastAPI()
handler = Mangum(app)

MODEL_PATH = "models/audio_model.keras"
IMG_SIZE = 125   # unused here, but you can remove
# load once at cold‑start
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded")
except Exception as e:
    print(f"Failed loading model: {e}")
    model = None

# --- audio utilities ---
def normalize_waveform(waveform: tf.Tensor) -> tf.Tensor:
    waveform = tf.cast(waveform, tf.float32)
    max_abs = tf.reduce_max(tf.abs(waveform))
    return waveform / (max_abs + 1e-6)

def get_spectrogram(waveform: tf.Tensor) -> tf.Tensor:
    spec = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spec = tf.abs(spec)
    # add channel axis
    return spec[..., tf.newaxis]

# --- inference endpoint ---
@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files supported")

    # read into TF
    contents = await file.read()
    try:
        audio_tensor, sample_rate = tf.audio.decode_wav(
            contents, desired_channels=1, desired_samples=16000
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid WAV file")

    # preprocess
    waveform = tf.squeeze(audio_tensor, axis=-1)
    waveform = normalize_waveform(waveform)
    spectrogram = get_spectrogram(waveform[tf.newaxis, :])

    # predict
    logits = model(spectrogram, training=False)[0]
    probs = tf.nn.softmax(logits).numpy()
    class_id = int(tf.argmax(logits).numpy())
    class_names = ['down','go','left','no','right','stop','up','yes']
    predicted = class_names[class_id]
    confidence = float(probs[class_id] * 100.0)

    return JSONResponse({
        "predicted_class": predicted,
        "confidence_percent": round(confidence, 2)
    })

@app.get("/")
def health():
    return {"status": "healthy"}
