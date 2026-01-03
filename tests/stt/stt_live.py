#!/usr/bin/env python3
"""
Live speech-to-text transcription using faster-whisper and sounddevice.
Captures audio from microphone, detects voice activity, and transcribes in real-time.
"""

import argparse
import sys
import time
import threading
import queue
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import librosa

# Defaults
DEFAULT_MODEL_DIR = "/data/models/Systran/faster-whisper-large-v3"
WHISPER_SAMPLE_RATE = 16000  # Whisper requires 16kHz
DEVICE_SAMPLE_RATE = 48000   # Most hardware uses 48kHz
BLOCK_DURATION = 0.5  # seconds per audio block
SILENCE_THRESHOLD = 0.005  # RMS threshold for silence detection
SILENCE_DURATION = 0.8  # seconds of silence before transcribing
MIN_AUDIO_DURATION = 0.5  # minimum audio duration to transcribe


def rms(audio: np.ndarray) -> float:
    """Calculate root mean square of audio signal."""
    return float(np.sqrt(np.mean(audio ** 2)))


def main():
    parser = argparse.ArgumentParser(description="Live speech-to-text transcription")
    parser.add_argument("--model", default=DEFAULT_MODEL_DIR, help="Path to whisper model")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--language", default=None, help="Language code (e.g., 'en', 'de'). Auto-detect if not set.")
    parser.add_argument("--threshold", type=float, default=SILENCE_THRESHOLD, help="Silence RMS threshold")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of CUDA")
    parser.add_argument("--sample-rate", type=int, default=DEVICE_SAMPLE_RATE, help="Input device sample rate (default 48000)")
    parser.add_argument("--debug", action="store_true", help="Show audio levels for debugging")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    print(f"[INIT] Loading model: {args.model}", flush=True)
    t0 = time.time()
    if args.cpu:
        model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=4)
        print(f"[INIT] Model loaded on CPU in {time.time() - t0:.2f}s", flush=True)
    else:
        try:
            model = WhisperModel(args.model, device="cuda", compute_type="int8_float16")
            print(f"[INIT] Model loaded on CUDA in {time.time() - t0:.2f}s", flush=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                print(f"[WARN] CUDA failed ({e}), falling back to CPU...", flush=True)
                t0 = time.time()
                model = WhisperModel(args.model, device="cpu", compute_type="int8", cpu_threads=4)
                print(f"[INIT] Model loaded on CPU in {time.time() - t0:.2f}s", flush=True)
            else:
                raise

    # Audio buffer and state
    audio_buffer = []
    silence_start = None
    is_speaking = False
    lock = threading.Lock()
    transcribe_queue = queue.Queue()

    input_sample_rate = args.sample_rate
    block_size = int(input_sample_rate * BLOCK_DURATION)

    debug_counter = [0]  # mutable for closure

    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_buffer, silence_start, is_speaking

        if status:
            print(f"[WARN] {status}", file=sys.stderr, flush=True)

        # Convert int16 to float32 normalized to [-1, 1]
        audio = indata[:, 0].astype(np.float32) / 32768.0
        level = rms(audio)
        peak = float(np.max(np.abs(audio)))

        # Debug output every ~1 second
        if args.debug:
            debug_counter[0] += 1
            if debug_counter[0] % int(1.0 / BLOCK_DURATION) == 0:
                print(f"\r[DEBUG] RMS: {level:.4f}, Peak: {peak:.4f}, Speaking: {is_speaking}    ", end="", flush=True)

        with lock:
            if level > args.threshold:
                # Voice detected
                audio_buffer.append(audio)
                silence_start = None
                if not is_speaking:
                    is_speaking = True
                    print("\r[...listening...]", end="", flush=True)
            else:
                # Silence
                if is_speaking:
                    audio_buffer.append(audio)  # include trailing silence
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > SILENCE_DURATION:
                        # Silence long enough, trigger transcription
                        if len(audio_buffer) > 0:
                            combined = np.concatenate(audio_buffer)
                            duration = len(combined) / input_sample_rate
                            if duration >= MIN_AUDIO_DURATION:
                                transcribe_queue.put(combined)
                        audio_buffer = []
                        silence_start = None
                        is_speaking = False

    def transcription_worker():
        detected_language = args.language
        while True:
            audio = transcribe_queue.get()
            if audio is None:
                break

            try:
                t0 = time.time()
                duration = len(audio) / input_sample_rate

                # Resample to 16kHz if needed
                if input_sample_rate != WHISPER_SAMPLE_RATE:
                    audio = librosa.resample(audio, orig_sr=input_sample_rate, target_sr=WHISPER_SAMPLE_RATE)

                # Detect language on first transcription if not specified
                if detected_language is None:
                    segments, info = model.transcribe(
                        audio,
                        language=None,
                        task="transcribe",
                        vad_filter=True,
                        beam_size=5
                    )
                    detected_language = info.language
                    print(f"\r[LANG] Detected: {detected_language}", flush=True)
                else:
                    segments, _ = model.transcribe(
                        audio,
                        language=detected_language,
                        task="transcribe",
                        vad_filter=True,
                        beam_size=5
                    )

                text_parts = []
                for segment in segments:
                    text_parts.append(segment.text.strip())

                text = " ".join(text_parts)
                elapsed = time.time() - t0

                if text:
                    print(f"\r[{duration:.1f}s -> {elapsed:.2f}s] {text}", flush=True)

            except Exception as e:
                print(f"\r[ERROR] Transcription failed: {e}", file=sys.stderr, flush=True)

    # Start transcription worker thread
    worker = threading.Thread(target=transcription_worker, daemon=True)
    worker.start()

    print(f"[READY] Listening on device {args.device or 'default'} @ {input_sample_rate}Hz... (Ctrl+C to stop)", flush=True)
    print(f"[INFO] Silence threshold: {args.threshold}, Language: {args.language or 'auto-detect'}", flush=True)
    print("-" * 60, flush=True)

    try:
        with sd.InputStream(
            samplerate=input_sample_rate,
            blocksize=block_size,
            device=args.device,
            channels=1,
            dtype=np.int16,  # Use int16, convert to float in callback
            callback=audio_callback
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user", flush=True)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # Cleanup
    try:
        transcribe_queue.put(None)  # Signal worker to stop
        worker.join(timeout=2)
    except Exception:
        pass
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
