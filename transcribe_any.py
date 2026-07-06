import os
import sys
import subprocess
import whisper
import time


def log(msg):
    print(msg, flush=True)
    with open("transcribe.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}\n")


def extract_audio(input_file, output_file):
    ffmpeg_path = os.path.join("tools", "ffmpeg", "ffmpeg.exe")
    if not os.path.exists(ffmpeg_path):
        raise FileNotFoundError(f"[FATAL] ffmpeg not found at {ffmpeg_path}")

    cmd = [ffmpeg_path, "-y", "-i", input_file, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1", output_file]
    log(f"[INFO] Running ffmpeg: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log(f"[ERROR] ffmpeg failed: {result.stderr}")
        raise RuntimeError("Audio extraction failed")
    log("[INFO] Audio extracted successfully.")


def transcribe_audio(audio_file, output_file):
    log("[INFO] Loading Whisper model...")
    model = whisper.load_model("base")
    log("[INFO] Transcribing audio...")
    result = model.transcribe(audio_file)
    text = result.get("text", "")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)
    log(f"[SUCCESS] Transcription complete. Output saved to {output_file}")


def main():
    try:
        if len(sys.argv) != 2:
            log("Usage: python transcribe_any.py <input_audio_or_video_file>")
            return

        input_file = sys.argv[1]
        if not os.path.isfile(input_file):
            log(f"[ERROR] File not found: {input_file}")
            return

        audio_file = "temp_audio.wav"
        transcript_file = "transcript.txt"

        extract_audio(input_file, audio_file)
        transcribe_audio(audio_file, transcript_file)

    except Exception as e:
        log(f"[FATAL ERROR] {e}")


if __name__ == "__main__":
    main()