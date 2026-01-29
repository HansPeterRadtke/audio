import os
import onnxruntime as ort
import numpy as np
import soundfile as sf
import json

print("[DEBUG] Starting minimal Kokoro ONNX test")

try:
  model_dir = "/home/hans/dev/GPT/models/kokoro_82m_onnx"
  config_path = os.path.join(model_dir, "config.json")
  model_path = os.path.join(model_dir, "model.onnx")

  print("[DEBUG] Loading config from:", config_path)
  with open(config_path, 'r') as f:
    config = json.load(f)

  print("[DEBUG] Loading ONNX model from:", model_path)
  session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

  text = "This is a test."
  print("[DEBUG] Using placeholder token IDs")
  dummy_input = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64)

  print("[DEBUG] Running inference")
  audio = session.run(None, {"input": dummy_input})[0].squeeze()

  output_path = "output.wav"
  sf.write(output_path, audio, config.get("sampling_rate", 22050))
  print("[DEBUG] Audio saved to", output_path)

except Exception as e:
  print("[ERROR]", str(e))

print("[DEBUG] Done")