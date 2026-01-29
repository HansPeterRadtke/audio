import sounddevice           as sd
import numpy                 as np
import queue
import time

print("[DEBUG] Starting headset_interface.py")

# Settings
SAMPLE_RATE    = 44100
CHANNELS       = 1
DELAY_SECONDS  = 0.5
BUFFER_SIZE    = int(SAMPLE_RATE * DELAY_SECONDS)

print(f"[DEBUG] Sample rate     : {SAMPLE_RATE}")
print(f"[DEBUG] Channels        : {CHANNELS}")
print(f"[DEBUG] Delay (seconds) : {DELAY_SECONDS}")
print(f"[DEBUG] Buffer size     : {BUFFER_SIZE}")

# Buffer for delayed playback
audio_buffer = queue.Queue(maxsize=BUFFER_SIZE)

# Fill buffer initially with silence
for _ in range(BUFFER_SIZE):
  audio_buffer.put(0.0)

def audio_callback(indata, outdata, frames, time_info, status):
  try:
    if status:
      print(f"[DEBUG] Status: {status}")

    for i in range(frames):
      sample = indata[i][0] if len(indata[i]) > 0 else 0.0
      audio_buffer.put(sample)
      delayed_sample = audio_buffer.get()
      outdata[i][0] = delayed_sample

  except Exception as e:
    print(f"[DEBUG] Exception in callback: {e}")
    outdata.fill(0.0)

try:
  with sd.Stream(samplerate=SAMPLE_RATE,
                 channels=CHANNELS,
                 dtype='float32',
                 callback=audio_callback):
    print("[DEBUG] Audio stream started. Press Ctrl+C to stop.")
    while True:
      time.sleep(1)

except KeyboardInterrupt:
  print("[DEBUG] Stopped by user")

except Exception as e:
  print(f"[DEBUG] Exception: {e}")

print("[DEBUG] headset_interface.py finished")