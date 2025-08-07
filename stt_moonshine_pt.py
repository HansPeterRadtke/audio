import os
import traceback
import numpy as np
import librosa
import time
import onnxruntime as ort
from transformers import AutoTokenizer, AutoConfig

print("[DEBUG] Starting import", flush=True)
start = time.time()

MODEL_ID = "usefulsensors/moonshine-base"
MODEL_DIR = f"/home/hans/dev/GPT/models/{MODEL_ID}"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
config = AutoConfig.from_pretrained(MODEL_DIR)

encoder_path = os.path.join(MODEL_DIR, "encoder_model_quantized.onnx")
decoder_path = os.path.join(MODEL_DIR, "decoder_model_merged_quantized.onnx")

encoder_session = ort.InferenceSession(encoder_path)
decoder_session = ort.InferenceSession(decoder_path)

print(f"[DEBUG] Model and tokenizer loaded ({time.time() - start:.3f}s)", flush=True)

AUDIO_FILE = "/home/hans/dev/GPT/github/audio/data/test_audio.mp3"

try:
  start = time.time()
  print("[DEBUG] Loading audio", flush=True)
  audio, sr = librosa.load(AUDIO_FILE, sr=16000)
  print(f"[DEBUG] Audio loaded with librosa ({time.time() - start:.3f}s)", flush=True)

  print("[DEBUG] Chunking audio", flush=True)
  chunk_duration = 30
  chunk_samples = chunk_duration * sr
  chunks = [audio[i:i+chunk_samples] for i in range(0, len(audio), chunk_samples)]

  print(f"[DEBUG] Starting transcription of {len(chunks)} chunks", flush=True)

  for i, chunk in enumerate(chunks):
    print(f"[DEBUG] Chunk {i+1}/{len(chunks)}", flush=True)
    # Placeholder: actual inference using encoder_session and decoder_session
    print("[WARNING] Moonshine inference not yet implemented", flush=True)

except Exception as e:
  print("[ERROR]", str(e), flush=True)
  traceback.print_exc()