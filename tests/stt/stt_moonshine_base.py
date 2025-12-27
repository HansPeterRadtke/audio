import os
import traceback
import numpy as np
import librosa
import time
import onnxruntime as ort
from transformers import AutoTokenizer, AutoConfig

print("[DEBUG] Starting import", flush=True)
start_all = time.time()

MODEL_DIR = "/home/hans/dev/GPT/models/onnx-community/moonshine-base-ONNX"

encoder_path = os.path.join(MODEL_DIR, "onnx/encoder_model_quantized.onnx")
decoder_path = os.path.join(MODEL_DIR, "onnx/decoder_model_merged_quantized.onnx")
audio_path = "/home/hans/dev/GPT/data/test01.m4a"

try:
  tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
  config    = AutoConfig.from_pretrained(MODEL_DIR)

  encoder_session = ort.InferenceSession(encoder_path)
  decoder_session = ort.InferenceSession(decoder_path)
  print("[DEBUG] Model and tokenizer loaded", flush=True)

  print("[DEBUG] Loading audio", flush=True)
  audio, sr = librosa.load(audio_path, sr=16000)
  print("[DEBUG] Audio loaded", flush=True)

  chunk_duration = 30
  chunk_samples = chunk_duration * sr
  chunks = [audio[i:i+chunk_samples] for i in range(0, len(audio), chunk_samples)]
  print(f"[DEBUG] Total chunks: {len(chunks)}", flush=True)

  def generate(input_features, decoder, config):
    batch_size = input_features.shape[0]
    num_heads  = 8
    dim_kv     = 52
    num_layers = 8
    encoder_seq_len = input_features.shape[1]
    max_len    = min((input_features.shape[-2] // 2), config.max_position_embeddings)

    input_ids = np.array([[config.decoder_start_token_id]], dtype=np.int64)
    generated = input_ids

    past_key_values = {
      f"past_key_values.{layer}.decoder.key":   np.zeros((batch_size, num_heads, 0, dim_kv), dtype=np.float32)
      for layer in range(num_layers)
    }
    past_key_values.update({
      f"past_key_values.{layer}.decoder.value": np.zeros((batch_size, num_heads, 0, dim_kv), dtype=np.float32)
      for layer in range(num_layers)
    })
    past_key_values.update({
      f"past_key_values.{layer}.encoder.key":   np.zeros((batch_size, num_heads, encoder_seq_len, dim_kv), dtype=np.float32)
      for layer in range(num_layers)
    })
    past_key_values.update({
      f"past_key_values.{layer}.encoder.value": np.zeros((batch_size, num_heads, encoder_seq_len, dim_kv), dtype=np.float32)
      for layer in range(num_layers)
    })

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

  for i, chunk in enumerate(chunks):
    print(f"[DEBUG] Chunk {i+1}/{len(chunks)}", flush=True)
    try:
      input_values = np.expand_dims(chunk.astype(np.float32), axis=0)
      encoded = encoder_session.run(None, {"input_values": input_values})[0]

      print(f"[DEBUG] Encoder output shape: {encoded.shape}", flush=True)
      if encoded.ndim != 3:
        raise ValueError(f"Unexpected encoder output shape: {encoded.shape}")

      token_ids = generate(encoded, decoder_session, config)
      text = tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
      print(f"[TRANSCRIPT] {text}", flush=True)

    except Exception as e:
      print(f"[ERROR] Exception during chunk {i+1}: {e}", flush=True)
      traceback.print_exc()

except Exception as e:
  print("[ERROR] Unhandled exception", flush=True)
  traceback.print_exc()

print(f"[DEBUG] Total time: {time.time() - start_all:.3f}s", flush=True)