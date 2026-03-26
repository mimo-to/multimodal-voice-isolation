# ================================================================
# COLAB QUICKSTART  —  run these cells in order
# ================================================================

# ---- Cell 1: Mount Drive + install deps ----
from google.colab import drive
drive.mount('/content/drive')

# !pip install yt-dlp librosa soundfile -q

# ---- Cell 2: Clone / upload your repo ----
# If your repo is already on Drive, just set the path below.
# Otherwise, upload the files via the file browser on the left.
import sys
sys.path.insert(0, '/content/drive/MyDrive/PE2_Project/multimodal-voice-isolation')

# ---- Cell 3: Prepare data (run once) ----
# Option A — you have raw .mp4 files already on Drive
from prepare_data import prepare_from_local_videos
prepare_from_local_videos(
    video_dir  = '/content/drive/MyDrive/PE2_Project/raw_videos',
    output_dir = '/content/drive/MyDrive/PE2_Project/processed_segments',
)

# Option B — download YouTube Shorts
# from prepare_data import prepare_from_urls
# prepare_from_urls(
#     url_list   = ['https://www.youtube.com/shorts/XXXXX', ...],
#     output_dir = '/content/drive/MyDrive/PE2_Project/processed_segments',
# )

# ---- Cell 4: Test architecture ----
# !python test_run.py

# ---- Cell 5: Train ----
# !python train.py

# ---- Cell 6: Run inference on a test clip ----
# !python inference.py \
#     --video      /content/drive/MyDrive/PE2_Project/raw_videos/speaker.mp4 \
#     --noisy      /content/drive/MyDrive/PE2_Project/raw_videos/speaker.mp4 \
#     --output     /content/isolated.wav \
#     --checkpoint /content/drive/MyDrive/PE2_Project/checkpoints/cocktail_net_best.pth
