import time
start_all = time.time()
t0 = time.time()
print("[DEBUG] Starting import", flush=True)

import os
print(f"[DEBUG] os imported ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()
import traceback
print(f"[DEBUG] traceback imported ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()
import numpy as np
print(f"[DEBUG] numpy imported ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()
import torchaudio
print(f"[DEBUG] torchaudio imported ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()
from faster_whisper import WhisperModel
print(f"[DEBUG] faster_whisper imported ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

print("[DEBUG] Finished all imports", flush=True)

try:
  model_dir = "/home/hans/dev/GPT/models/Systran/faster-whisper-large-v3"
  audio_path = "/home/hans/dev/GPT/data/test01.m4a"
  print(f"[DEBUG] Paths set ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

  if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model path does not exist: {model_dir}")
  if not os.path.isfile(audio_path):
    raise FileNotFoundError(f"Audio file not found: {audio_path}")
  print(f"[DEBUG] Path checks done ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

  model = WhisperModel(model_dir, compute_type="int8")
  print(f"[DEBUG] Model loaded ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

  waveform, sample_rate = torchaudio.load(audio_path)
  if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
  print(f"[DEBUG] Audio loaded & resampled ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

  audio_np = waveform.squeeze().numpy().astype(np.float32)
  print(f"[DEBUG] Converted to numpy ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

  chunk_size = 32000 * 2  # 2 seconds @ 16kHz
  num_chunks = len(audio_np) // chunk_size + (1 if len(audio_np) % chunk_size != 0 else 0)

  print(f"[DEBUG] Starting chunked transcription: {num_chunks} chunks", flush=True)
  for i in range(num_chunks):
    chunk = audio_np[i*chunk_size:(i+1)*chunk_size]
    chunk_start = i * 2.0
    try:
      segments, _ = model.transcribe(chunk, language="en", vad_filter=True, word_timestamps=True)
      for segment in segments:
        print(f"[{chunk_start + segment.start:.2f}s -> {chunk_start + segment.end:.2f}s] {segment.text}", flush=True)
    except Exception as e:
      print(f"[ERROR] Exception during chunk {i}: {e}", flush=True)
      traceback.print_exc()

except Exception as e:
  print("[ERROR] Unhandled exception", flush=True)
  traceback.print_exc()

print(f"[DEBUG] Total script time: {time.time() - start_all:.3f}s", flush=True)
print("[DEBUG] Script finished", flush=True)