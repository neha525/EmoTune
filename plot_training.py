"""
plot_training.py
----------------
Generates a single, clean training history graph.
Reads epoch-end metrics from training.log (UTF-16 LE) or other log fallbacks.

Run from speech_emotion_music/ folder:
    python plot_training.py
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

BASE   = Path(__file__).parent
MODELS = BASE / "models"

_ANSI = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
_PAT  = re.compile(
    r'accuracy:\s*([\d.]+)\s*-\s*loss:\s*([\d.]+)'
    r'\s*-\s*val_accuracy:\s*([\d.]+)\s*-\s*val_loss:\s*([\d.]+)'
)

def _try_read(path: Path) -> str:
    """Try multiple encodings; return clean (ANSI-stripped) text or None."""
    for enc in ("utf-16", "utf-16-le", "utf-8"):
        try:
            raw = path.read_text(encoding=enc, errors="ignore")
            return _ANSI.sub("", raw)
        except Exception:
            continue
    return None


def load_history():
    for log_name in ["training.log", "train_new.log", "train_run.log"]:
        p = BASE / log_name
        if not p.exists():
            continue
        clean = _try_read(p)
        if clean is None:
            continue
        acc, val_acc, loss, val_loss = [], [], [], []
        for m in _PAT.finditer(clean):
            a = float(m.group(1))
            # Skip batch-level lines: they share the same pattern but
            # only epoch-end lines have val_accuracy present.
            acc.append(a)
            loss.append(float(m.group(2)))
            val_acc.append(float(m.group(3)))
            val_loss.append(float(m.group(4)))
        if len(acc) >= 2:
            print(f"[INFO] Parsed {len(acc)} epoch records from {log_name}")
            return acc, val_acc, loss, val_loss
    raise FileNotFoundError(
        "No usable training log found. Please run train_model.py first."
    )


def plot(acc, val_acc, loss, val_loss, out: Path):
    epochs   = list(range(1, len(acc) + 1))
    best_ep  = int(np.argmax(val_acc)) + 1
    best_val = max(val_acc)

    # ── Palette ───────────────────────────────────────────────────────────────
    BG    = "#0d1117"
    PANEL = "#161b22"
    TRAIN = "#f78166"
    VAL   = "#58a6ff"
    GRID  = "#21262d"
    TEXT  = "#c9d1d9"
    BEST  = "#3fb950"
    SHADE = "#8b949e"

    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.facecolor":    PANEL,
        "figure.facecolor":  BG,
        "axes.edgecolor":    GRID,
        "axes.labelcolor":   TEXT,
        "xtick.color":       TEXT,
        "ytick.color":       TEXT,
        "grid.color":        GRID,
        "legend.facecolor":  PANEL,
        "legend.edgecolor":  GRID,
        "legend.labelcolor": TEXT,
        "text.color":        TEXT,
    })

    fig = plt.figure(figsize=(14, 5.5), facecolor=BG)
    fig.suptitle("Speech Emotion Recognition — Training History",
                 fontsize=15, fontweight="bold", color=TEXT, y=0.97)

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30,
                           left=0.07, right=0.97, top=0.87, bottom=0.12)

    def smooth(v, w=3):
        return np.convolve(v, np.ones(w) / w, mode="valid")

    def draw(ax, y_train, y_val, title, y_label, pct=False):
        yt = [v * 100 for v in y_train] if pct else list(y_train)
        yv = [v * 100 for v in y_val]   if pct else list(y_val)

        ax.plot(epochs, yt, color=TRAIN, lw=2.3, label="Train", alpha=0.9)
        ax.plot(epochs, yv, color=VAL,   lw=2.3, label="Val",   alpha=0.9)
        ax.fill_between(epochs, yt, yv, alpha=0.07, color=TEXT)

        if len(epochs) >= 5:
            s_ep = epochs[1:-1]
            ax.plot(s_ep, smooth(yt), color=TRAIN, lw=1.0, ls="--", alpha=0.5)
            ax.plot(s_ep, smooth(yv), color=VAL,   lw=1.0, ls="--", alpha=0.5)

        bv = yv[best_ep - 1]
        ax.axvline(best_ep, color=BEST, lw=1.4, ls="--", alpha=0.75)

        off_x = max(1, len(epochs) * 0.06)
        label = f"Best: {bv:.1f}{'%' if pct else ''}\n(ep {best_ep})"
        ax.annotate(label,
                    xy=(best_ep, bv),
                    xytext=(best_ep + off_x, bv - (7 if pct else bv * 0.05)),
                    color=BEST, fontsize=9,
                    arrowprops=dict(arrowstyle="->", color=BEST, lw=1.1))

        if pct:
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
            ax.set_ylim(0, 108)

        ax.set_title(title, color=TEXT, fontsize=12, pad=8)
        ax.set_xlabel("Epoch", color=TEXT)
        ax.set_ylabel(y_label, color=TEXT)
        ax.grid(True, lw=0.6, alpha=0.6)
        ax.legend(framealpha=0.6, fontsize=9.5)

    draw(fig.add_subplot(gs[0, 0]),
         acc, val_acc, "Accuracy", "Accuracy (%)", pct=True)
    draw(fig.add_subplot(gs[0, 1]),
         loss, val_loss, "Loss", "Loss")

    fig.text(
        0.5, 0.02,
        f"Epochs: {len(epochs)}   |   "
        f"Best val accuracy: {best_val*100:.2f}% (epoch {best_ep})   |   "
        f"Final train acc: {acc[-1]*100:.2f}%   |   "
        f"Final val acc: {val_acc[-1]*100:.2f}%",
        ha="center", va="center", fontsize=9, color=SHADE
    )

    plt.savefig(str(out), dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"\n✅  Graph saved  →  {out}")


if __name__ == "__main__":
    try:
        acc, val_acc, loss, val_loss = load_history()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    plot(acc, val_acc, loss, val_loss, MODELS / "accuracy_graph.png")
