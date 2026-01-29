import os
import sys
import json
import soundfile as sf
import numpy as np

print("[DEBUG] Script start")

try:
  model_dir = "/home/hans/dev/GPT/models/kokoro_82m_onnx"
  config_path = os.path.join(model_dir, "config.json")

  print("[DEBUG] Loading config:", config_path)
  with open(config_path, 'r') as f:
    config = json.load(f)

  print("[DEBUG] Loading model from model.py")
  sys.path.insert(0, model_dir)
  from model import KokoroONNX

  print("[DEBUG] Initializing model")
  model = KokoroONNX(config, model_dir)

  text = "This is a test of the Kokoro eighty two million parameter ONNX voice model."
  speaker = "default"
  print("[DEBUG] Synthesizing text:", text)

  audio = model.infer(text, speaker=speaker)

  output_path = "output.wav"
  sf.write(output_path, audio, config.get("sampling_rate", 24000))
  print("[DEBUG] Audio saved to", output_path)

except Exception as e:
  print("[ERROR]", str(e))

print("[DEBUG] Script end")