import json
import csv
import argparse
from pathlib import Path

print("[DEBUG] Starting create_train_data.py")

try:
  parser  = argparse.ArgumentParser(description="Generate training data manifest from WhisperX alignment JSON.")
  parser .add_argument("--alignment_json" , required=True, help="Path to WhisperX alignment.json file")
  parser .add_argument("--output_manifest", required=True, help="Path to output CSV manifest")
  parser .add_argument("--audio_path"     , required=True, help="Absolute path to the source audio file")
  args    = parser.parse_args()
except Exception as e:
  print(f"[DEBUG] Failed to parse arguments: {e}")
  exit(1)

alignment_path = Path(args.alignment_json)
manifest_path  = Path(args.output_manifest)
audio_path     = args.audio_path

print(f"[DEBUG] Alignment JSON  : {alignment_path}")
print(f"[DEBUG] Output Manifest : {manifest_path}")
print(f"[DEBUG] Audio Path      : {audio_path}")

try:
  with open(alignment_path, "r", encoding="utf-8") as f:
    data     = json.load(f)
    segments = data.get("segments", [])
except Exception as e:
  print(f"[DEBUG] Failed to read alignment JSON: {e}")
  exit(1)

try:
  with open(manifest_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["audio", "start", "end", "text"])
    for seg in segments:
      writer.writerow([audio_path, seg.get("start"), seg.get("end"), seg.get("text")])
  print("[DEBUG] Manifest successfully written.")
except Exception as e:
  print(f"[DEBUG] Failed to write manifest: {e}")
  exit(1)

print("[DEBUG] create_train_data.py finished")