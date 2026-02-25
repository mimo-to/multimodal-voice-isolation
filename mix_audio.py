"""
mix_audio.py — Mixes two mp4 videos into one noisy wav file (16kHz).
"""
import subprocess
import librosa
import numpy as np
import soundfile as sf
import os

RAW_DIR    = "./raw_videos"
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

videos = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.mp4')])
print("Available videos:")
for i, v in enumerate(videos):
    print(f"  [{i}] {v}")

TARGET_IDX = 0
NOISE_IDX  = 1

target_path = os.path.join(RAW_DIR, videos[TARGET_IDX])
noise_path  = os.path.join(RAW_DIR, videos[NOISE_IDX])

def extract_audio(video_path, out_wav, sr=16000):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ar", str(sr), "-ac", "1",
        "-t", "5",
        out_wav
    ], capture_output=True)

print(f"\nExtracting audio...")
extract_audio(target_path, f"{OUTPUT_DIR}/target_raw.wav")
extract_audio(noise_path,  f"{OUTPUT_DIR}/noise_raw.wav")

y_t, _ = librosa.load(f"{OUTPUT_DIR}/target_raw.wav", sr=16000)
y_n, _ = librosa.load(f"{OUTPUT_DIR}/noise_raw.wav",  sr=16000)

min_len = min(len(y_t), len(y_n))
y_mixed = (y_t[:min_len] * 0.5) + (y_n[:min_len] * 0.5)

sf.write(f"{OUTPUT_DIR}/mixed_input.wav",    y_mixed,       16000)
sf.write(f"{OUTPUT_DIR}/speaker1_clean.wav", y_t[:min_len], 16000)
sf.write(f"{OUTPUT_DIR}/speaker2_clean.wav", y_n[:min_len], 16000)

print(f"Mixed audio ready")
print(f"   Target : {videos[TARGET_IDX]}")
print(f"   Noise  : {videos[NOISE_IDX]}")