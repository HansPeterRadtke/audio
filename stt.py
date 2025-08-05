import os
import traceback
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

print("[DEBUG] Starting STT script")

try:
  model_dir = "/home/hans/dev/GPT/models/whisper-base"
  audio_path = "/home/hans/dev/GPT/data/REC00001.mp3"

  print(f"[DEBUG] Checking if model path exists: {model_dir}")
  if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model directory not found: {model_dir}")

  print(f"[DEBUG] Checking if audio file exists: {audio_path}")
  if not os.path.isfile(audio_path):
    raise FileNotFoundError(f"Audio file not found: {audio_path}")

  os.environ["TRANSFORMERS_OFFLINE"] = "1"
  os.environ["HF_DATASETS_OFFLINE"] = "1"

  print("[DEBUG] Loading processor and model from local directory")
  processor = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
  model     = WhisperForConditionalGeneration.from_pretrained(model_dir, local_files_only=True)
  model     .eval()

  device = "cuda" if torch.cuda.is_available() else "cpu"
  model  = model.to(device)

  print("[DEBUG] Loading and resampling audio")
  speech_array, sampling_rate = torchaudio.load(audio_path)
  if sampling_rate != 16000:
    resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
    speech_array = resampler(speech_array)

  input_features = processor(speech_array[0], sampling_rate=16000, return_tensors="pt").input_features.to(device)

  print("[DEBUG] Generating transcription")
  predicted_ids = model.generate(input_features)
  transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

  print("[DEBUG] Transcription result:")
  print(transcription)

except Exception as e:
  print("[ERROR] An error occurred:")
  traceback.print_exc()

print("[DEBUG] STT script finished")