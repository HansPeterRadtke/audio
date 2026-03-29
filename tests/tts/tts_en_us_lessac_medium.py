import os
import json
import numpy as np
import soundfile as sf
import onnxruntime as ort
import traceback

print("[DEBUG] Running Piper TTS test")

try:
  model_dir = "/home/hans/dev/GPT/models/kitten_tts"
  model_path = os.path.join(model_dir, "en_US-lessac-medium.onnx")
  config_path = os.path.join(model_dir, "en_US-lessac-medium.onnx.json")

  # Load config
  print("[DEBUG] Loading config")
  with open(config_path, "r") as f:
    config = json.load(f)

  phoneme_map = config["phoneme_id_map"]
  sample_rate = config["audio"]["sample_rate"]

  # Prepare input text
  text = "hello world"
  ids = []
  for char in text:
    if char in phoneme_map:
      ids.append(phoneme_map[char][0])
    else:
      print(f"[WARN] Skipping unknown char: {repr(char)}")

  input_array = np.array([ids], dtype=np.int64)
  input_lengths = np.array([len(ids)], dtype=np.int64)
  scales = np.array([1.0, 0.667, 0.8], dtype=np.float32)  # FIXED

  print("[DEBUG] Input IDs:", ids)
  print("[DEBUG] Shape:", input_array.shape)

  session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

  audio = session.run(None, {
    "input": input_array,
    "input_lengths": input_lengths,
    "scales": scales
  })[0]

  output_path = "/home/hans/dev/GPT/github/audio/output.wav"
  sf.write(output_path, audio.squeeze(), sample_rate)
  print("[SUCCESS] Audio written to:", output_path)

except Exception as e:
  print("[ERROR] Exception occurred:", e)
  traceback.print_exc()