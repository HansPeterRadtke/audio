import os
import onnxruntime as ort
import numpy as np
import soundfile as sf
import yaml

print("[INFO] Starting NeuML LJSpeech VITS ONNX TTS script")

try:
  from ttstokenizer import TTSTokenizer
except ImportError as e:
  print("[ERROR] ttstokenizer module not found! Please install it with: pip install ttstokenizer")
  raise e

try:
  model_dir = "/home/hans/dev/GPT/models/NeuML_ljspeech-vits-onnx"
  model_path = os.path.join(model_dir, "model.onnx")
  config_path = os.path.join(model_dir, "config.yaml")

  print("[INFO] Loading config from:", config_path)
  with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

  print("[INFO] Loading model from:", model_path)
  session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

  print("[INFO] Initializing tokenizer")
  tokenizer = TTSTokenizer(config["token"]["list"])

  text = "Hello. This is a test of the NeuML LJSpeech VITS ONNX voice model."
  print("[INFO] Text:", text)

  print("[INFO] Tokenizing text")
  input_ids = tokenizer(text)

  print("[INFO] Running inference...")
  audio = session.run(None, {"text": input_ids})[0].squeeze()

  output_path = "output.wav"
  sf.write(output_path, audio, config.get("sample_rate", 22050))
  print("[INFO] Audio saved to", output_path)

except Exception as e:
  print("[ERROR]", str(e))

print("[INFO] Done.")