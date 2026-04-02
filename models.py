import os
import shutil

import numpy as np
import torch
import torch.serialization
from dotenv import load_dotenv

load_dotenv()

_FFMPEG_FALLBACK = (
    r"C:\Users\rouna\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-8.1-full_build\bin\ffmpeg.exe"
)

_ffmpeg_env = os.getenv("FFMPEG_PATH", "").strip()
FFMPEG = (
    _ffmpeg_env if _ffmpeg_env
    else shutil.which("ffmpeg") or (
        _FFMPEG_FALLBACK if os.path.isfile(_FFMPEG_FALLBACK) else None
    )
)
if not FFMPEG:
    raise RuntimeError("ffmpeg not found. Set FFMPEG_PATH in .env or run: winget install Gyan.FFmpeg")

TARGET_SR = 16000
SEP_MODEL = None
SEP_BACKEND = "spectral"

SEP_MODEL_REPO = os.getenv("SEP_MODEL_REPO", "JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")


def _setup_hf_auth():
    token = os.getenv("HF_TOKEN", "")
    if token:
        os.environ["HF_TOKEN"] = token
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass


def _allow_asteroid_pickle():
    try:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except Exception:
        pass


def load_separator():
    global SEP_MODEL, SEP_BACKEND, TARGET_SR

    _setup_hf_auth()
    _allow_asteroid_pickle()

    try:
        from asteroid.models import ConvTasNet
    except ImportError:
        print("asteroid not installed, using spectral fallback")
        return

    candidates = [
        (SEP_MODEL_REPO, 16000),
        ("JorisCos/ConvTasNet_Libri2Mix_sepclean_8k", 8000),
    ]

    for repo, sr in candidates:
        try:
            print(f"loading {repo}...")
            model = ConvTasNet.from_pretrained(repo)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            SEP_MODEL = model
            SEP_BACKEND = "convtasnet"
            TARGET_SR = sr
            print(f"conv-tasnet ready ({sr // 1000}kHz)")
            return
        except Exception as e:
            print(f"  {repo} failed: {e}")

    print("all conv-tasnet checkpoints failed, falling back to spectral")
