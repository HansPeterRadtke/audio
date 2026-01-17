import os
import sys
import argparse
import traceback
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import time

DEFAULT_MODEL = "openai/whisper-base"
DEFAULT_AUDIO = "/data/tmp/test_100_words.wav"

start_all = time.time()
print("[DEBUG] Starting STT script")

try:
  parser = argparse.ArgumentParser(description="Whisper STT using HuggingFace transformers")
  parser.add_argument("audio", nargs="?", default=DEFAULT_AUDIO, help="Audio file path")
  parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or path")
  args = parser.parse_args()

  t0 = time.time()
  model_id = args.model
  audio_path = args.audio
  print(f"[DEBUG] Model: {model_id}, Audio: {audio_path}")

  t0 = time.time()
  if not os.path.isfile(audio_path):
    raise FileNotFoundError(f"Audio file not found: {audio_path}")
  print(f"[DEBUG] Path checks done ({time.time() - t0:.3f}s)")

  t0 = time.time()
  processor = WhisperProcessor.from_pretrained(model_id)
  model     = WhisperForConditionalGeneration.from_pretrained(model_id)
  model.eval()
  print(f"[DEBUG] Model load complete ({time.time() - t0:.3f}s)")

  t0 = time.time()
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model  = model.to(device)
  print(f"[DEBUG] Model moved to device ({time.time() - t0:.3f}s)")

  t0 = time.time()
  speech_array, sampling_rate = torchaudio.load(audio_path)
  if sampling_rate != 16000:
    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    speech_array = resampler(speech_array)
  print(f"[DEBUG] Audio load & resample ({time.time() - t0:.3f}s)")

  t0 = time.time()
  input_features = processor(speech_array[0], sampling_rate=16000, return_tensors="pt").input_features.to(device)
  print(f"[DEBUG] Input feature prep ({time.time() - t0:.3f}s)")

  t0 = time.time()
  predicted_ids = model.generate(input_features)
  transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
  print(f"[DEBUG] Transcription complete ({time.time() - t0:.3f}s)")

  print("[DEBUG] Transcription result:")
  print(transcription)

except Exception as e:
  print("[ERROR] An error occurred:")
  traceback.print_exc()

print(f"[DEBUG] Total script time: {time.time() - start_all:.3f}s")
print("[DEBUG] STT script finished")