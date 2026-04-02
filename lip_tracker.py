import os
import urllib.request

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models"
    "/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# central pair weighted higher, outer pairs stabilise
LIP_PAIRS = [(13, 14), (12, 15), (11, 16)]
LIP_WEIGHTS = [0.5, 0.25, 0.25]

SMOOTH_WINDOW = int(os.getenv("LIP_SMOOTH_WINDOW", "5"))


def _ensure_model():
    if not os.path.isfile(_MODEL_PATH):
        print("downloading face landmarker model (~30MB)...")
        try:
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
            print("face landmarker ready")
        except Exception as e:
            print(f"model download failed: {e}")
            return False
    return True


def _build_detector():
    if not _ensure_model():
        return None
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=_MODEL_PATH),
        num_faces=1,
        min_face_detection_confidence=0.15,
        min_face_presence_confidence=0.15,
        min_tracking_confidence=0.15,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def get_lip_movement(video_path, max_seconds=None):
    if max_seconds is None:
        max_seconds = int(os.getenv("LIP_MAX_SECONDS", "5"))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"cannot open: {video_path}")
        return {"signal": np.array([0.0]), "fps": 30.0, "detection_rate": 0.0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(fps * max_seconds)

    detector = _build_detector()
    if detector is None:
        cap.release()
        return {"signal": np.array([0.0]), "fps": fps, "detection_rate": 0.0}

    raw_gaps = []
    detected = 0
    prev_gap = 0.0

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        if w < 320:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            gap = sum(
                wt * abs(lm[lo].y - lm[hi].y)
                for (hi, lo), wt in zip(LIP_PAIRS, LIP_WEIGHTS)
            )
            prev_gap = gap
            detected += 1
        else:
            gap = prev_gap

        raw_gaps.append(gap)

    cap.release()
    detector.close()

    if not raw_gaps:
        return {"signal": np.array([0.0]), "fps": fps, "detection_rate": 0.0}

    rate = detected / len(raw_gaps)
    print(f"lip detection: {rate:.0%} over {len(raw_gaps)} frames")

    signal = np.array(raw_gaps, dtype=np.float64)

    if len(signal) > SMOOTH_WINDOW:
        signal = uniform_filter1d(signal, size=SMOOTH_WINDOW)

    lo, hi = signal.min(), signal.max()
    signal = (signal - lo) / (hi - lo) if hi > lo else np.zeros_like(signal)

    return {"signal": signal, "fps": fps, "detection_rate": rate}


def plot_lip_movement(lip_data, out_path=None):
    signal = lip_data["signal"]
    fps = lip_data["fps"]
    times = np.arange(len(signal)) / fps

    fig, ax = plt.subplots(figsize=(13, 3))
    ax.plot(times, signal, color="royalblue", linewidth=1.5, label="Lip activity")
    ax.axhline(np.mean(signal), color="crimson", linestyle="--", linewidth=1,
               label=f"Mean ({np.mean(signal):.3f})")
    ax.fill_between(times, signal, alpha=0.15, color="royalblue")
    ax.set_title("Lip Activity Signal", fontsize=11)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalised gap")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(times[0], times[-1])
    fig.tight_layout()
    path = out_path or os.path.join(OUTPUT_DIR, "lip_movement.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "./raw_videos/test.mp4"
    data = get_lip_movement(path)
    print(f"{len(data['signal'])} frames  max={data['signal'].max():.4f}  std={data['signal'].std():.4f}")
    plot_lip_movement(data)
