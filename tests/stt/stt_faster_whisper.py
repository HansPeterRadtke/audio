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
import librosa
print(f"[DEBUG] librosa imported ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()
from faster_whisper import WhisperModel
print(f"[DEBUG] faster_whisper imported ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

print("[DEBUG] Finished all imports", flush=True)

try:
  default_audio_path = "/home/hans/dev/GPT/data/test01.m4a"
  model_dir          = "/srv/data/models/Systran/faster-whisper-large-v3"

  audio_path = os.sys.argv[1] if len(os.sys.argv) > 1 else default_audio_path
  print(f"[DEBUG] Audio path set to: {audio_path}", flush=True)

  if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model path does not exist: {model_dir}")
  if not os.path.isfile(audio_path):
    raise FileNotFoundError(f"Audio file not found: {audio_path}")
  print(f"[DEBUG] Path checks done ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

  model = WhisperModel(model_dir, compute_type="int8", cpu_threads=4)
  print(f"[DEBUG] Model loaded ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

  audio, sr = librosa.load(audio_path, sr=16000)
  print(f"[DEBUG] Audio loaded with librosa ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

  txt_path = os.path.splitext(audio_path)[0] + ".txt"
  try:
    output_file = open(txt_path, "w", encoding="utf-8")
    print(f"[DEBUG] Created output file: {txt_path}", flush=True)
  except Exception as e:
    print(f"[ERROR] Failed to create output file: {txt_path}", flush=True)
    traceback.print_exc()
    raise

  detect_samples   = min(len(audio), 30 * 16000)
  _, info = model.transcribe(audio[:detect_samples], language=None, task="transcribe", vad_filter=True)
  lang = info.language
  print(f"[DEBUG] Detected language: {lang} ({getattr(info,'language_probability',None)})", flush=True)

  chunk_duration = 30
  chunk_samples  = chunk_duration * 16000
  total_samples  = len(audio)
  num_chunks     = total_samples // chunk_samples + (1 if total_samples % chunk_samples != 0 else 0)

  print(f"[DEBUG] Starting chunked transcription: {num_chunks} chunks", flush=True)

  for i in range(num_chunks):
    start_sample = i * chunk_samples
    end_sample   = min(start_sample + chunk_samples, total_samples)
    chunk        = audio[start_sample:end_sample].astype(np.float32)
    chunk_start  = i * chunk_duration

    try:
      segments, _ = model.transcribe(chunk, language=lang, task="transcribe", vad_filter=True, word_timestamps=True)
      for segment in segments:
        output_file.write(segment.text + "\n")
        output_file.flush()
        print(f"[{chunk_start + segment.start:.2f}s -> {chunk_start + segment.end:.2f}s] {segment.text}", flush=True)
    except Exception as e:
      print(f"[ERROR] Exception during chunk {i}: {e}", flush=True)
      traceback.print_exc()

  try:
    output_file.close()
    print(f"[DEBUG] Output file closed: {txt_path}", flush=True)
  except Exception as e:
    print(f"[ERROR] Failed to close output file: {txt_path}", flush=True)
    traceback.print_exc()

except Exception as e:
  print("[ERROR] Unhandled exception", flush=True)
  traceback.print_exc()

print(f"[DEBUG] Total script time: {time.time() - start_all:.3f}s", flush=True)
print("[DEBUG] Script finished", flush=True)
