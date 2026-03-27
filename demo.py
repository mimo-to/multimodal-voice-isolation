"""
demo.py — Full pipeline: mix → separate → lip track → match → play
"""
import librosa
import numpy as np
import soundfile as sf
import os, sys, subprocess

OUTPUT_DIR = "./outputs"
RAW_DIR    = "./raw_videos"

def voice_energy(audio, sr=8000, window=0.1):
    win_samples = int(sr * window)
    return np.array([
        np.sqrt(np.mean(audio[i:i+win_samples]**2))
        for i in range(0, len(audio) - win_samples, win_samples)
    ])

def play(path):
    abs_path = os.path.abspath(path)
    if sys.platform == "win32":
        os.startfile(abs_path)
    elif sys.platform == "darwin":
        subprocess.call(["afplay", abs_path])
    else:
        subprocess.call(["ffplay", "-nodisp", "-autoexit", abs_path])

def run_demo():
    print("\n" + "="*55)
    print("  COCKTAIL PARTY PROBLEM — FULL DEMO")
    print("="*55)

    print("\n[1/4] Creating noisy mix...")
    os.system("python mix_audio.py")

    print("\n[2/4] Separating voices...")
    os.system("python separate.py")

    print("\n[3/4] Analyzing lip movement...")
    os.system("python lip_tracker.py")

    print("\n[4/4] Matching voice to lips...")
    v1, _ = librosa.load(f"{OUTPUT_DIR}/separated_voice1.wav", sr=8000)
    v2, _ = librosa.load(f"{OUTPUT_DIR}/separated_voice2.wav", sr=8000)

    from lip_tracker import get_lip_movement
    videos = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.mp4')])
    times, lips = get_lip_movement(os.path.join(RAW_DIR, videos[0]))

    e1 = voice_energy(v1)
    e2 = voice_energy(v2)
    lip_resampled = np.interp(
        np.linspace(0, len(lips)-1, len(e1)),
        np.arange(len(lips)), lips
    )

    corr1 = np.corrcoef(lip_resampled, e1)[0, 1]
    corr2 = np.corrcoef(lip_resampled, e2)[0, 1]
    print(f"\n   Voice 1 ↔ Lips correlation: {corr1:.3f}")
    print(f"   Voice 2 ↔ Lips correlation: {corr2:.3f}")

    matched = f"{OUTPUT_DIR}/separated_voice1.wav" if corr1 >= corr2 \
              else f"{OUTPUT_DIR}/separated_voice2.wav"
    other   = f"{OUTPUT_DIR}/separated_voice2.wav" if corr1 >= corr2 \
              else f"{OUTPUT_DIR}/separated_voice1.wav"

    winner = "Voice 1" if corr1 >= corr2 else "Voice 2"
    print(f"   ✅ {winner} matched to target speaker's lips")

    print("\n" + "="*55)
    print("  RESULTS")
    print("="*55)

    print("\n▶  Playing MIXED INPUT (noisy)...")
    play(f"{OUTPUT_DIR}/mixed_input.wav")

    input("\nPress Enter to hear ISOLATED target speaker...")
    play(matched)

    input("\nPress Enter to hear the OTHER speaker...")
    play(other)

    print("\n✅ Demo complete!")
    print(f"   All audio → {OUTPUT_DIR}/")
    print(f"   Lip chart → {OUTPUT_DIR}/lip_movement.png")

if __name__ == "__main__":
    run_demo()