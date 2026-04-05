"""
run_train.py
Helper launcher — sets env vars then calls train() directly.
Logs everything to train_run.log (tee to stdout too).
"""
import os, sys

os.environ["PYTHONUTF8"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Redirect stdout + stderr to log file (and keep tee to console)
log_path = os.path.join(os.path.dirname(__file__), "train_new.log")
log_fh = open(log_path, "w", encoding="utf-8", buffering=1)

class Tee:
    def __init__(self, *targets): self.targets = targets
    def write(self, data):
        for t in self.targets:
            try: t.write(data)
            except Exception: pass
    def flush(self):
        for t in self.targets:
            try: t.flush()
            except Exception: pass
    def fileno(self): return self.targets[0].fileno()

sys.stdout = Tee(sys.__stdout__, log_fh)
sys.stderr = Tee(sys.__stderr__, log_fh)

# Import and run
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
from train_model import train

train(
    dataset_dir="dataset/ravdess",
    epochs=60,
    augment=True,
    label_smoothing=0.1,
    use_attention=True,
    save_path=Path("models/ser_model.h5"),
)
