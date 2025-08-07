import os
import traceback
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import time

start_all = time.time()
print("[DEBUG] Starting STT script")

try:
  t0 = time.time()
  model_dir = "/home/hans/dev/GPT/models/whisper-base"
  audio_path = "/home/hans/dev/GPT/data/test01.m4a"
  print(f"[DEBUG] Paths set ({time.time() - t0:.3f}s)")

  t0 = time.time()
  if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory not found: {model_dir}")
  if not os.path.isfile(audio_path):
    raise FileNotFoundError(f"Audio file not found: {audio_path}")
  print(f"[DEBUG] Path checks done ({time.time() - t0:.3f}s)")

  os.environ["TRANSFORMERS_OFFLINE"] = "1"
  os.environ["HF_DATASETS_OFFLINE"] = "1"

  t0 = time.time()
  processor = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
  model     = WhisperForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
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