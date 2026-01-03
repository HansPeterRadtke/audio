#!/usr/bin/env python3
import argparse
import sys
import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

def main():
    parser = argparse.ArgumentParser(description="Direct Piper TTS using Python library - no files")
    parser.add_argument("--model", required=True, help="Full path to .onnx model")
    parser.add_argument("--text", default=None, help="Text to speak")
    parser.add_argument("--text-file", default=None, help="File with text to speak")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    voice = PiperVoice.load(args.model)

    if args.text:
        text = args.text
    elif args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    if not text.strip():
        print("No text provided")
        return

    print("Synthesizing and playing...")
    
    # It's a property, not a method
    audio_list = []
    for chunk in voice.synthesize(text):
        audio_list.append(chunk.audio_int16_array)
    
    # Concatenate and play
    audio_data = np.concatenate(audio_list)
    print(f"Audio length: {len(audio_data)} samples, {len(audio_data)/voice.config.sample_rate:.2f} seconds")
    sd.play(audio_data, voice.config.sample_rate, blocking=True)

    print("Done.")

if __name__ == "__main__":
    main()