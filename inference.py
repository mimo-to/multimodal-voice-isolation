"""
inference.py
------------
Given a video file of a target speaker mixed with background voices,
produce an isolated audio file.

Usage:
    python inference.py --video speaker.mp4 --noisy mixed.wav --output clean.wav
    python inference.py --video speaker.mp4 --noisy speaker.mp4 --output clean.wav
"""

import os
import argparse
import numpy as np
import cv2
import librosa
import torch

from models.fusion_net import CocktailNet
from utils.audio_processor import reconstruct_audio, SR, N_FFT, HOP_LENGTH

# match training constants
NUM_VIDEO_FRAMES = 50
NUM_AUDIO_FRAMES = 200
IMG_SIZE = 112

VIDEO_MEAN = np.array([0.43216, 0.39466, 0.37645], dtype=np.float32)
VIDEO_STD  = np.array([0.22803, 0.22145, 0.21699], dtype=np.float32)


def load_video_tensor(video_path):
    """Read NUM_VIDEO_FRAMES frames from a video file and return a model-ready tensor."""
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # uniformly sample NUM_VIDEO_FRAMES from the full clip
    indices = set(np.linspace(0, total_frames - 1, NUM_VIDEO_FRAMES, dtype=int))

    frames = []
    for i in range(total_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if i in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame.astype(np.float32) / 255.0
            frame = (frame - VIDEO_MEAN) / VIDEO_STD
            frames.append(frame)
        if len(frames) == NUM_VIDEO_FRAMES:
            break
    cap.release()

    while len(frames) < NUM_VIDEO_FRAMES:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))

    video = np.stack(frames, axis=0)               # [T, H, W, C]
    video = torch.FloatTensor(video).permute(3, 0, 1, 2)  # [C, T, H, W]
    return video.unsqueeze(0)                       # [1, C, T, H, W]


def load_audio_tensor(audio_path):
    """Load audio, compute STFT magnitude, return model-ready tensor."""
    y, _ = librosa.load(audio_path, sr=SR)

    target_len = SR * 2
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = np.abs(stft)  # [257, T]

    if mag.shape[1] < NUM_AUDIO_FRAMES:
        mag = np.pad(mag, ((0, 0), (0, NUM_AUDIO_FRAMES - mag.shape[1])))
    else:
        mag = mag[:, :NUM_AUDIO_FRAMES]

    return torch.FloatTensor(mag).unsqueeze(0)  # [1, 257, 200]


def run_inference(video_path, noisy_audio_path, output_path, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load model
    model = CocktailNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    # support both raw state_dict and checkpoint dict
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # prepare inputs
    video_t = load_video_tensor(video_path).to(device)     # [1, 3, 50, 112, 112]
    audio_t = load_audio_tensor(noisy_audio_path).to(device)  # [1, 257, 200]

    # predict mask
    with torch.no_grad():
        mask = model(video_t, audio_t)  # [1, 257, 200]

    predicted_mask = mask[0]  # [257, 200]

    # apply mask to noisy audio and save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    reconstruct_audio(noisy_audio_path, predicted_mask, output_path)
    print(f"Isolated audio saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      required=True, help="Path to target speaker video")
    parser.add_argument("--noisy",      required=True, help="Path to noisy/mixed audio (wav or mp4)")
    parser.add_argument("--output",     default="output/isolated.wav")
    parser.add_argument("--checkpoint", default="/content/drive/MyDrive/PE2_Project/checkpoints/cocktail_net_best.pth")
    args = parser.parse_args()

    run_inference(args.video, args.noisy, args.output, args.checkpoint)


