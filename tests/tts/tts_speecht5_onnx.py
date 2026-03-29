from txtai.pipeline import TextToSpeech
import soundfile as sf

# Load the ONNX TTS model
tts = TextToSpeech("/home/hans/dev/GPT/models/txtai-speecht5-onnx")

# Run TTS
audio_array, rate = tts("Hello, I am an offline TTS model!")

# Save output to Explorer folder
output_path = "/var/www/html/explorer/upload/output.wav"
sf.write(output_path, audio_array, rate)
print("[SUCCESS] Audio written to:", output_path)