print("[DEBUG] Script started")
#!/usr/bin/env python3

import os
import sys
import yaml
import onnxruntime
import soundfile as sf
import numpy as np

try:
  from g2p_en import G2p
except ImportError:
  print("[ERROR] g2p_en not installed. Run: pip install g2p_en")
  sys.exit(1)

# Simple tokenizer implementation
class SimpleTokenizer:
  def __init__(self, token_list):
    self.token_to_id = {tok: idx for idx, tok in enumerate(token_list)}
    self.unk_id = self.token_to_id.get("<unk>", 1)

  def __call__(self, tokens):
    ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]
    return np.array(ids, dtype=np.int64)


def main():
  print("[DEBUG] Starting TTS with ljspeech-jets-onnx model (English text + G2P)")

  model_dir = "/home/hans/dev/GPT/models/ljspeech-jets-onnx"
  config_path = os.path.join(model_dir, "config.yaml")
  model_path = os.path.join(model_dir, "model.onnx")

  if not os.path.exists(config_path):
    print(f"[ERROR] Missing config file: {config_path}")
    sys.exit(1)
  if not os.path.exists(model_path):
    print(f"[ERROR] Missing model file: {model_path}")
    sys.exit(1)

  # Load config
  with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

  token_list = config["token"]["list"]
  tokenizer = SimpleTokenizer(token_list)
  g2p = G2p()

  text = (
    "This is test 987654321. "
    + 
    "This is a test. <laugh> Let's see how it handles special tags like <breath> and <noise>. "
    "<sos/eos> Hello world. This is a test. I am trying to make it sound expressive! "
    "Have you ever heard of someone laugh? Now I am laughing <sos/eos>. "
    "That's funny. Now I am going to breathe <sos/eos>. "
    "We are trying to simulate a breath. Okay, now let's try a question? "
    "What is your name? I am excited to speak long sentences! "
    "Let's try an exclamation! Oh wow! "
    "This is great. We can use multiple punctuation marks.. "
    "You can hear the differences. Now I am going to say goodbye! "
    "I want to pause for a moment ... and then continue. "
    "Sometimes people say ah... or hmm... before thinking. "
    "Do you understand? This should test the intonation. "
    "Let's add one more exclamation! Amazing! "
    "Finally, thank you for listening. Goodbye."
  )

  print("Input text:", text)

  phonemes = g2p(text)
  print("Phoneme count:" , len(phonemes))
  print("Phonemes:", phonemes)

  if len(phonemes) > 512:
    print(f"[WARNING] Truncating phonemes from {len(phonemes)} to 512")
    phonemes = phonemes[:512]

  input_ids = tokenizer(phonemes)
  print("[DEBUG] Input shape:", input_ids.shape)

  sess = onnxruntime.InferenceSession(model_path)
  outputs = sess.run(None, {"text": input_ids})
  wav = outputs[0].squeeze()
  print("[DEBUG] Output waveform shape:", wav.shape)

  output_path = os.path.join(model_dir, "output.wav")
  sf.write(output_path, wav, samplerate=22050)
  print("[DEBUG] Saved to:", output_path)

  try:
    os.system(f"cp -v {output_path} /var/www/html/explorer/upload/output.wav")
  except Exception as e:
    print("[ERROR] Could not copy to explorer folder:", e)

if __name__ == "__main__":
  main()