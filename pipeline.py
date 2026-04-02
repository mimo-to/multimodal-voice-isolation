import os
import subprocess
import traceback
import time
import threading

import numpy as np
import librosa
import soundfile as sf
import torch
from dotenv import load_dotenv

import models

load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
MAX_CLIP_SECONDS = int(os.getenv("MAX_CLIP_SECONDS", "10"))

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

JOBS: dict = {}


def normalize(audio, headroom_db=-1.0):
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    return (audio / peak) * (10 ** (headroom_db / 20))


def extract_audio(src, dst, duration=None):
    if duration is None:
        duration = MAX_CLIP_SECONDS
    subprocess.run(
        [models.FFMPEG, "-y", "-i", src,
         "-ar", str(models.TARGET_SR), "-ac", "1",
         "-t", str(duration), dst],
        capture_output=True, check=True,
    )


def mix_tracks(path_a, path_b, out_path):
    a, _ = librosa.load(path_a, sr=models.TARGET_SR)
    b, _ = librosa.load(path_b, sr=models.TARGET_SR)
    n = min(len(a), len(b))
    mixed = a[:n] * 0.5 + b[:n] * 0.5
    sf.write(out_path, mixed, models.TARGET_SR)
    return n / models.TARGET_SR


def separate(mixed_path, out1, out2):
    sr = models.TARGET_SR
    if models.SEP_BACKEND == "convtasnet":
        data, file_sr = sf.read(mixed_path, always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if file_sr != sr:
            data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
        data = normalize(data)

        device = next(models.SEP_MODEL.parameters()).device
        tensor = torch.tensor(data).unsqueeze(0).unsqueeze(0).to(device)

        with torch.inference_mode():
            est = models.SEP_MODEL(tensor)

        sf.write(out1, normalize(est[0, 0].cpu().numpy()), sr)
        sf.write(out2, normalize(est[0, 1].cpu().numpy()), sr)
    else:
        y, _ = librosa.load(mixed_path, sr=sr, mono=True)
        S = librosa.stft(y, n_fft=2048, hop_length=512)
        mag = np.abs(S)
        # per-bin median mask — better than temporal average but still rough
        mask = (mag > np.median(mag, axis=1, keepdims=True)).astype(np.float32)
        sf.write(out1, normalize(librosa.istft(S * mask, hop_length=512)), sr)
        sf.write(out2, normalize(librosa.istft(S * (1 - mask), hop_length=512)), sr)


def rms_envelope(audio, fps):
    win = max(1, int(models.TARGET_SR / fps))
    frames = len(audio) // win
    if frames == 0:
        return np.array([0.0])
    blocks = audio[:frames * win].reshape(frames, win)
    return np.sqrt(np.mean(blocks ** 2, axis=1))


def zscore(x):
    s = np.std(x)
    if s < 1e-8:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s


def match_speaker(lips, detection_rate, e1, e2, fps):
    n = min(len(lips), len(e1), len(e2))
    e1, e2 = e1[:n], e2[:n]

    if detection_rate < 0.05 or np.std(lips[:n]) < 1e-6:
        total = e1.mean() + e2.mean() + 1e-9
        return round(float(e1.mean() / total), 3), round(float(e2.mean() / total), 3), "energy"

    ref = zscore(lips[:n])
    z1 = zscore(e1)
    z2 = zscore(e2)

    c1 = float(np.corrcoef(ref, z1)[0, 1])
    c2 = float(np.corrcoef(ref, z2)[0, 1])
    c1 = 0.0 if np.isnan(c1) else round(c1, 3)
    c2 = 0.0 if np.isnan(c2) else round(c2, 3)
    return c1, c2, "pearson"


def log(job, msg):
    job.setdefault("logs", []).append(msg)


def run_pipeline(job_id, v1_path, v2_path):
    job = JOBS[job_id]
    prefix = os.path.join(OUTPUT_DIR, job_id)

    try:
        job.update(step="mixing", progress=5)
        log(job, "extracting audio...")
        extract_audio(v1_path, f"{prefix}_t.wav")
        extract_audio(v2_path, f"{prefix}_n.wav")

        duration = mix_tracks(f"{prefix}_t.wav", f"{prefix}_n.wav", f"{prefix}_mixed.wav")
        log(job, f"mixed {duration:.1f}s of audio")
        job.update(progress=20)

        job.update(step="separating", progress=22)
        label = "Conv-TasNet" if models.SEP_BACKEND == "convtasnet" else "Spectral"
        log(job, f"source separation ({label})...")
        separate(f"{prefix}_mixed.wav", f"{prefix}_v1.wav", f"{prefix}_v2.wav")
        log(job, "separation done")
        job.update(progress=68)

        job.update(step="lip_tracking", progress=70)
        log(job, "tracking lips...")
        from lip_tracker import get_lip_movement, plot_lip_movement
        lip_data = get_lip_movement(v1_path, max_seconds=int(os.getenv("LIP_MAX_SECONDS", "5")))

        face_ok = lip_data["detection_rate"] >= 0.05
        if face_ok:
            log(job, f"face tracked: {lip_data['detection_rate']:.0%} over {len(lip_data['signal'])} frames")
            plot_lip_movement(lip_data, out_path=f"{prefix}_lips.png")
        else:
            log(job, "no face detected, using energy-based matching")
        job.update(progress=84)

        job.update(step="matching", progress=86)
        log(job, "correlating...")

        v1_audio, _ = librosa.load(f"{prefix}_v1.wav", sr=models.TARGET_SR)
        v2_audio, _ = librosa.load(f"{prefix}_v2.wav", sr=models.TARGET_SR)

        fps = lip_data["fps"]
        e1 = rms_envelope(v1_audio, fps)
        e2 = rms_envelope(v2_audio, fps)

        c1, c2, method = match_speaker(
            lip_data["signal"], lip_data["detection_rate"], e1, e2, fps
        )

        matched = "voice1" if c1 >= c2 else "voice2"
        log(job, f"track 1: {c1:.3f}  track 2: {c2:.3f} ({method})")
        log(job, f"target: {matched}")

        job.update(
            step="done", progress=100, status="complete",
            corr1=c1, corr2=c2,
            matched=matched,
            method=method,
            duration=round(duration, 1),
            voice1=f"{job_id}_v1.wav",
            voice2=f"{job_id}_v2.wav",
            mixed=f"{job_id}_mixed.wav",
            graph_url=f"/api/image/{job_id}_lips.png" if face_ok else None,
        )

    except Exception as e:
        job.update(status="error", error=str(e), tb=traceback.format_exc())
        log(job, f"error: {e}")


def cleanup_old_files(max_age_seconds=3600):
    now = time.time()
    count = 0
    for folder in [UPLOAD_DIR, OUTPUT_DIR]:
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            if os.path.isfile(path) and (now - os.path.getmtime(path)) > max_age_seconds:
                try:
                    os.remove(path)
                    count += 1
                except OSError:
                    pass
    if count:
        print(f"cleanup: removed {count} old files")


def start_cleanup_thread(interval=None):
    if interval is None:
        interval = int(os.getenv("CLEANUP_INTERVAL_SECONDS", "1800"))

    def loop():
        while True:
            time.sleep(interval)
            cleanup_old_files()

    threading.Thread(target=loop, daemon=True).start()
