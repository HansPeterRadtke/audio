import os
import numpy as np
import onnxruntime as ort
import soundfile as sf
import yaml
from ttstokenizer import TTSTokenizer

print("[DEBUG] Starting NeuML LJSpeech JETS ONNX TTS test")

try:
  model_dir = "/home/hans/dev/GPT/models/ljspeech_jets_onnx"
  model_path = os.path.join(model_dir, "model.onnx")
  config_path = os.path.join(model_dir, "config.yaml")

  print("[DEBUG] Loading config:", config_path)
  with open(config_path, "r") as f:
    config = yaml.safe_load(f)

  print("[DEBUG] Initializing tokenizer")
  tokenizer = TTSTokenizer(config["token"]["list"])

  text = (
    "Well, I must say, it's quite fascinating to witness how these systems have evolved."
    " Not too long ago, synthetic voices sounded robotic and lifeless."
    " But now, they're filled with nuance, rhythm, and even a hint of emotion — don't you agree?"
  )

  print("[DEBUG] Tokenizing text")
  tokens = tokenizer(text)
  print("[DEBUG] Tokens:", tokens.tolist())

  print("[DEBUG] Initializing ONNX runtime")
  session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

  print("[DEBUG] Running inference")
  audio = session.run(None, {"text": tokens})[0].squeeze()

  output_path = "/var/www/html/explorer/upload/output.wav"
  sf.write(output_path, audio, config.get("sample_rate", 22050))
  print("[DEBUG] Audio written to", output_path)

except Exception as e:
  print("[ERROR]", str(e))

print("[DEBUG] Done")