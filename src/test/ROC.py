"""
ROC Curve Generator
===================
Computes and plots ROC curves (One-vs-Rest) for train, validation, and test sets.

Folder structure expected:
    Final-DATA/KNEE-main-images-stablized/train/<class>/images...
    Final-DATA/KNEE-main-images-stablized/val/<class>/images...
    Final-DATA/KNEE-main-images-stablized/test/<class>/images...

Requirements:
    pip install tensorflow numpy matplotlib scikit-learn pillow
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tensorflow import keras
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR   = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
MODEL_PATH = BASE_DIR / "model/final-model.keras"

SETS = {
    "Train":      BASE_DIR / "Final-DATA/KNEE-main-images-stablized/train",
    "Validation": BASE_DIR / "Final-DATA/KNEE-main-images-stablized/val",
    "Test":       BASE_DIR / "Final-DATA/KNEE-main-images-stablized/test",
}

IMG_SIZE    = 224
CLASS_NAMES = ["0", "1", "2", "3", "4"]
N_CLASSES   = len(CLASS_NAMES)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_dataset(folder: Path):
    """Load all images and labels from a folder of class subfolders."""
    images, labels = [], []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = folder / class_name
        if not class_dir.exists():
            print(f"  [warn] folder not found: {class_dir}")
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
                images.append(np.array(img, dtype=np.float32) / 255.0)
                labels.append(class_idx)
            except Exception as e:
                print(f"  [warn] skipping {img_path.name}: {e}")

    return np.array(images), np.array(labels)


def plot_roc(y_true_bin, y_score, set_name, ax):
    """Plot per-class and macro-average ROC on a given axis."""
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

    macro_tpr = np.linspace(0, 1, 300)
    tprs      = []

    for i in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc     = auc(fpr, tpr)
        interp_tpr  = np.interp(macro_tpr, fpr, tpr)
        tprs.append(interp_tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=1.5,
                label=f"Class {CLASS_NAMES[i]} (AUC = {roc_auc:.3f})")

    # Macro average
    mean_tpr      = np.mean(tprs, axis=0)
    mean_tpr[0]   = 0.0
    macro_auc     = auc(macro_tpr, mean_tpr)
    ax.plot(macro_tpr, mean_tpr, color="black", lw=2.5, linestyle="--",
            label=f"Macro Average (AUC = {macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curve — {set_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading model ...")
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded.\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("ROC Curves — Knee Osteoarthritis Classification",
                 fontsize=15, fontweight="bold", y=1.02)

    for ax, (set_name, set_path) in zip(axes, SETS.items()):
        print(f"Processing {set_name} set: {set_path}")
        X, y = load_dataset(set_path)
        print(f"  Loaded {len(X)} images")

        print(f"  Running predictions ...")
        y_score = model.predict(X, batch_size=32, verbose=0)
        if isinstance(y_score, list):
            y_score = y_score[0]

        y_true_bin = label_binarize(y, classes=list(range(N_CLASSES)))
        plot_roc(y_true_bin, y_score, set_name, ax)
        print(f"  Done.\n")

    plt.tight_layout()
    out_path = BASE_DIR / "roc_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()