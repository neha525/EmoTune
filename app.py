"""
app.py
------
Flask web server for the Speech Emotion Recognition & Music Recommendation system.

Endpoints
---------
  GET  /                  → Single-page web UI
  POST /predict           → { wav: <base64 or file upload> } → { emotion, probs, songs, session }
  POST /predict-file      → multipart WAV upload
  GET  /history           → last 20 sessions (JSON)
  POST /clear-history     → wipe history.json
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS

# Force UTF-8 I/O — guard against child-process scenarios where stdout is not a real TTY
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# ── File-based error logging (captures tracebacks even when PowerShell hides stderr)
_LOG_FILE = Path("flask_error.log")
_file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
_file_handler.setLevel(logging.ERROR)
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
app.logger.addHandler(_file_handler)
logging.getLogger("werkzeug").addHandler(_file_handler)

# Lazy-load heavy modules once on first request
_predictor = None



def get_predictor():
    global _predictor
    if _predictor is None:
        from emotion_predictor import EmotionPredictor
        _predictor = EmotionPredictor()
    return _predictor


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html") 

@app.route("/app")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept a WAV file (multipart OR base64 JSON) and return predictions."""
    tmp_path = None
    try:
        # ── Obtain audio bytes and detect extension ───────────────────────
        ext = ".wav"
        if request.content_type and "multipart" in request.content_type:
            if "audio" not in request.files:
                return jsonify({"error": "No audio file in request"}), 400
            file_obj = request.files["audio"]
            filename  = file_obj.filename or "audio.wav"
            ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ".wav"
            wav_bytes = file_obj.read()
        else:
            data = request.get_json(force=True) or {}
            b64 = data.get("audio")
            if not b64:
                return jsonify({"error": "No audio data provided"}), 400
            if "," in b64:
                b64 = b64.split(",", 1)[1]
            wav_bytes = base64.b64decode(b64)

        # ── Write to temp file so librosa can read it (keep ext for codec) ─
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
            tf.write(wav_bytes)
            tmp_path = tf.name

        # ── Predict ─────────────────────────────────────────────────────────
        predictor = get_predictor()
        label, probs = predictor.predict_from_file(tmp_path)

        # ── Recommend ───────────────────────────────────────────────────────
        from music_recommendation import recommend_music
        songs = recommend_music(label, probs, mode="auto", top_n=5)

        # ── Build prob_dict with correct class order ─────────────────────────
        probs_list  = probs.tolist() if hasattr(probs, "tolist") else list(probs)
        class_order = predictor._classes   # ['angry','fear','happy','neutral','sad','surprise']
        prob_dict   = {emo: round(float(p), 4) for emo, p in zip(class_order, probs_list)}

        # ── Log session (pass prob_dict so history uses correct keys) ────────
        from session_logger import log_session
        session_entry = log_session(label, prob_dict, songs)

        return jsonify({
            "emotion":  label,
            "probs":    prob_dict,
            "songs":    songs,
            "session":  session_entry,
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@app.route("/history", methods=["GET"])
def history():
    try:
        from session_logger import load_history
        n = request.args.get("n", 20, type=int)
        return jsonify(load_history(n))
    except Exception as exc:
        tb = traceback.format_exc()
        app.logger.error("[/history] %s\n%s", exc, tb)
        _LOG_FILE.write_text(tb, encoding="utf-8", errors="replace")
        return jsonify({"error": str(exc), "traceback": tb}), 500


@app.route("/clear-history", methods=["POST"])
def clear_history_route():
    from session_logger import clear_history
    clear_history()
    return jsonify({"status": "cleared"})


# ─────────────────────────────────────────────────────────────────────────────
# Global error handler — catches anything not handled by route try/except
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(Exception)
def handle_exception(exc):
    tb = traceback.format_exc()
    try:
        _LOG_FILE.write_text(f"[GLOBAL HANDLER]\n{tb}", encoding="utf-8")
    except Exception:
        pass
    app.logger.error("[GLOBAL HANDLER] %s\n%s", exc, tb)
    return jsonify({"error": str(exc), "traceback": tb}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Dev server entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SER Web App")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    try:
        print(f"\nSER Web App  ->  http://{args.host}:{args.port}/\n")
    except UnicodeEncodeError:
        print(f"\nSER Web App  ->  http://{args.host}:{args.port}/\n".encode("ascii", "replace").decode())
    app.run(host=args.host, port=args.port, debug=args.debug)
