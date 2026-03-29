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
  model_dir          = "/data/models/Systran/faster-whisper-large-v3"

  audio_path = os.sys.argv[1] if len(os.sys.argv) > 1 else default_audio_path
  print(f"[DEBUG] Audio path set to: {audio_path}", flush=True)

  if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model path does not exist: {model_dir}")
  if not os.path.isfile(audio_path):
    raise FileNotFoundError(f"Audio file not found: {audio_path}")
  print(f"[DEBUG] Path checks done ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

  total_sec = float(librosa.get_duration(path=audio_path))
  print(f"[DEBUG] AudioDuration = %6.0f sec = %6.2f min; ({time.time() - t0:.3f}s)" % (total_sec, (total_sec / 60)), flush=True); t0 = time.time()

  model = WhisperModel(model_dir, compute_type="int8", cpu_threads=2)
  print(f"[DEBUG] Model loaded ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

  txt_path = os.path.splitext(audio_path)[0] + ".txt"
  try:
    output_file = open(txt_path, "w", encoding="utf-8")
    print(f"[DEBUG] Created output file: {txt_path}", flush=True)
  except Exception as e:
    print(f"[ERROR] Failed to create output file: {txt_path}", flush=True)
    traceback.print_exc()
    raise

  try:
    print("Transcribing ...", flush = True)
    t0 = time.time()
    segments, info = model.transcribe(audio_path                    ,
                                      language        = None        ,
                                      task            = "transcribe",
                                      vad_filter      = True        ,
                                      word_timestamps = True        ,
                                      chunk_length    = 15          ,
                                      beam_size       =  1          ,
                                      condition_on_previous_text = False)
    td = (time.time() - t0)
    print("DONE; Language is '%s'; Took %6.0f sec = %6.2f min; " % (info.language, td, (td / 60)))
    for seg in segments:
      print("Writing segment [%7.1f ... %7.1f]; %6.2f %%; " % (seg.start, seg.end, ((seg.end / total_sec) * 100)), flush = True)
      output_file.write(seg.text + "\n")
      output_file.flush()
      print("%s" % (seg.text), flush = True)
  except Exception as e:
    print(f"[ERROR] Exception during segment-iteration: {e}", flush=True)
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
