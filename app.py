import os
import uuid
import threading

from flask import Flask, request, jsonify, send_file, abort, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

from models import load_separator
from pipeline import JOBS, UPLOAD_DIR, OUTPUT_DIR, run_pipeline, start_cleanup_thread

load_dotenv()

load_separator()
start_cleanup_thread()

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_MB", "100")) * 1024 * 1024

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route("/api/process", methods=["POST"])
def process():
    if "video1" not in request.files or "video2" not in request.files:
        return jsonify(error="both video1 and video2 are required"), 400

    for key in ("video1", "video2"):
        f = request.files[key]
        if not f.content_type or not f.content_type.startswith("video/"):
            return jsonify(error=f"{key} must be a video file"), 415
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(0)
        if size > MAX_FILE_SIZE:
            return jsonify(error=f"{key} exceeds {os.getenv('MAX_FILE_MB', '100')}MB limit"), 413

    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"status": "running", "step": "queued", "progress": 0, "logs": []}

    v1 = os.path.join(UPLOAD_DIR, f"{job_id}_v1.mp4")
    v2 = os.path.join(UPLOAD_DIR, f"{job_id}_v2.mp4")
    request.files["video1"].save(v1)
    request.files["video2"].save(v2)

    JOBS[job_id]["logs"].append("upload complete, starting pipeline...")
    threading.Thread(target=run_pipeline, args=(job_id, v1, v2), daemon=True).start()
    return jsonify(job_id=job_id), 202


@app.route("/api/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="job not found"), 404
    return jsonify(job), 200


@app.route("/api/audio/<filename>")
def serve_audio(filename):
    path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype="audio/wav")


@app.route("/api/image/<filename>")
def serve_image(filename):
    return send_from_directory(OUTPUT_DIR, os.path.basename(filename))


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=False, port=port, threaded=True, use_reloader=False)
