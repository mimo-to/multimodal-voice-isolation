# Voice Isolation Pipeline

A multimodal cocktail party solver. Given two videos of people speaking simultaneously, the system mixes their audio, runs neural source separation (Conv-TasNet) to produce two isolated tracks, then uses MediaPipe lip tracking to correlate each track's energy envelope against the target speaker's lip movement. The track with the highest Pearson correlation is returned as the isolated voice.

---

## Setup

```bash
git clone <repo-url>
cd image-voice-isolation

python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt

cp .env.example .env
# edit .env and set HF_TOKEN
```

Install ffmpeg if not already on PATH:

```bash
winget install Gyan.FFmpeg
```

Run:

```bash
python app.py
```

Open `http://localhost:5000` in a browser.

---

## File Structure

```
image-voice-isolation/
├── app.py                  flask routes
├── models.py               model loading and config
├── pipeline.py             audio processing, separation, correlation
├── lip_tracker.py          mediapipe face landmark tracking
├── face_landmarker.task    mediapipe model (auto-downloaded on first run)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── outputs/                separated audio and lip graphs (auto-cleaned after 1h)
├── uploads/                uploaded video files (temp)
├── pretrained_models/      huggingface model cache
├── raw_videos/             place test videos here
├── .env                    local config (not committed)
├── .env.example            config template
└── requirements.txt
```

---

## Known Limitations

- Separation quality depends on Conv-TasNet loading successfully. If the model fails to download, the system falls back to a naive spectral mask which produces poor results.
- Lip tracking requires a clearly visible face in the target video. Low resolution, side angles, or heavy occlusion cause detection failure and fall back to energy-based matching.
- The Pearson correlation between lip movement and vocal energy is a weak signal. Both tracks may score similarly, especially when speakers overlap heavily or talk at the same time.
- Job state is in-memory only. All jobs are lost on server restart.
- The pipeline clips both videos to the first 10 seconds (configurable via `MAX_CLIP_SECONDS`).
- Flask development server only. Not suitable for concurrent users or production use.
