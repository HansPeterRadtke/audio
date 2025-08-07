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
import librosa
print(f"[DEBUG] librosa imported ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()
import onnxruntime as ort
print(f"[DEBUG] onnxruntime imported ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()
from transformers import AutoTokenizer, AutoConfig
print(f"[DEBUG] transformers imported ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

print("[DEBUG] Finished all imports", flush=True)

try:
  model_dir  = "/home/hans/dev/GPT/models/onnx-community/moonshine-base-ONNX"
  onnx_dir   = os.path.join(model_dir, "onnx")
  audio_path = "/home/hans/dev/GPT/data/test01.m4a"
  print(f"[DEBUG] Paths set ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

  if not os.path.exists(model_dir):
    raise FileNotFoundError(f"Model path does not exist: {model_dir}")
  if not os.path.isdir(onnx_dir):
    raise FileNotFoundError(f"ONNX path does not exist: {onnx_dir}")
  if not os.path.isfile(audio_path):
    raise FileNotFoundError(f"Audio file not found: {audio_path}")
  print(f"[DEBUG] Path checks done ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

  tokenizer = AutoTokenizer.from_pretrained(model_dir)
  config    = AutoConfig   .from_pretrained(model_dir)
  print(f"[DEBUG] Tokenizer and config loaded ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

  encoder = ort.InferenceSession(os.path.join(onnx_dir, "encoder_model_quantized.onnx"))
  decoder = ort.InferenceSession(os.path.join(onnx_dir, "decoder_model_merged_quantized.onnx"))
  print(f"[DEBUG] ONNX sessions loaded ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

  audio, sr = librosa.load(audio_path, sr=16000)
  print(f"[DEBUG] Audio loaded with librosa ({time.time() - t0:.3f}s)", flush=True); t0 = time.time()

  chunk_duration = 30
  chunk_samples  = chunk_duration * sr
  total_samples  = len(audio)
  num_chunks     = total_samples // chunk_samples + (1 if total_samples % chunk_samples != 0 else 0)

  print(f"[DEBUG] Starting chunked transcription: {num_chunks} chunks", flush=True)

  def generate(input_features, decoder, config):
    batch_size = input_features.shape[0]
    num_heads  = config.decoder_num_key_value_heads
    dim_kv     = config.hidden_size // config.decoder_num_attention_heads
    num_layers = config.decoder_num_hidden_layers
    max_len    = min((input_features.shape[-2] // 2), config.max_position_embeddings)

    input_ids = np.array([[config.decoder_start_token_id]], dtype=np.int64)
    generated = input_ids

    past_key_values = {
      f"past_key_values.{layer}.{branch}.{kv}": np.zeros((batch_size, num_heads, 0, dim_kv), dtype=np.float32)
      for layer in range(num_layers)
      for branch in ("decoder", "encoder")
      for kv in ("key", "value")
    }

    for step in range(max_len):
      decoder_inputs = {
        "input_ids": generated[:, -1:],
        "encoder_hidden_states": input_features,
        "use_cache_branch": np.array([step > 0], dtype=bool),
        **past_key_values
      }

      outputs = decoder.run(None, decoder_inputs)
      logits = outputs[0]
      next_token = logits[:, -1].argmax(-1, keepdims=True)
      generated = np.concatenate([generated, next_token], axis=-1)

      past_names = [x.name for x in decoder.get_outputs()][1:]
      for name, value in zip(past_names, outputs[1:]):
        past_key_values[name] = value

      if (next_token == config.eos_token_id).all():
        break

    return generated

  for i in range(num_chunks):
    start_sample = i * chunk_samples
    end_sample   = min(start_sample + chunk_samples, total_samples)
    chunk        = audio[start_sample:end_sample].astype(np.float32)

    try:
      print(f"[DEBUG] Chunk {i+1}/{num_chunks}", flush=True)
      encoded = encoder.run(None, {"input_values": np.expand_dims(chunk, axis=0)})[0]
      token_ids = generate(encoded, decoder, config)
      decoded = tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
      print(f"[TRANSCRIPT] {decoded}", flush=True)

    except Exception as e:
      print(f"[ERROR] Exception during chunk {i}: {e}", flush=True)
      traceback.print_exc()

except Exception as e:
  print("[ERROR] Unhandled exception", flush=True)
  traceback.print_exc()

print(f"[DEBUG] Total script time: {time.time() - start_all:.3f}s", flush=True)
print("[DEBUG] Script finished", flush=True)