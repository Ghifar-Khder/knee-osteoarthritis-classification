"""
FID Report Generator
====================
Computes Fréchet Inception Distance (FID) between real and GAN-generated images
for each class folder under:
    Final-DATA/KNEE-main-images-stablized/train/<class_folder>/

Images whose filename starts with "gan" are treated as generated;
all other images are treated as real.

Requirements:
    pip install torch torchvision scipy numpy Pillow tqdm
"""

import os
import re
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.linalg import sqrtm

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

BASE_DIR = Path("Final-DATA/KNEE-main-images-stablized/train")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH    = 32          # images per forward pass (lower if OOM)
IMG_SIZE = 299         # Inception v3 expects 299×299

# ── Helpers ──────────────────────────────────────────────────────────────────

TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

def load_image(path: Path) -> torch.Tensor | None:
    """Open an image, convert to RGB, apply transform. Returns None on error."""
    try:
        img = Image.open(path).convert("RGB")
        return TRANSFORM(img)
    except Exception as e:
        print(f"  [warn] skipping {path.name}: {e}")
        return None


def build_inception() -> torch.nn.Module:
    """Return Inception v3 with the final FC replaced by Identity (pool features)."""
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.aux_logits = False          # disable auxiliary head
    model.eval()
    return model.to(DEVICE)


@torch.no_grad()
def get_activations(paths: list[Path], model: torch.nn.Module) -> np.ndarray:
    """Extract 2048-dim Inception pool features for a list of image paths."""
    all_feats = []
    for i in range(0, len(paths), BATCH):
        batch_paths = paths[i : i + BATCH]
        tensors = [load_image(p) for p in batch_paths]
        tensors = [t for t in tensors if t is not None]
        if not tensors:
            continue
        batch = torch.stack(tensors).to(DEVICE)
        feats = model(batch)                      # (N, 2048)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0) if all_feats else np.array([])


def compute_fid(act_real: np.ndarray, act_fake: np.ndarray) -> float:
    """
    FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2·sqrt(Σ_r·Σ_f))
    Returns float('inf') when there are too few samples.
    """
    if len(act_real) < 2 or len(act_fake) < 2:
        return float("inf")

    mu_r, sigma_r = act_real.mean(0),  np.cov(act_real, rowvar=False)
    mu_f, sigma_f = act_fake.mean(0),  np.cov(act_fake, rowvar=False)

    diff = mu_r - mu_f
    # Matrix square-root (may have tiny imaginary parts → clip)
    covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))
    return fid


def split_images(folder: Path) -> tuple[list[Path], list[Path]]:
    """Return (real_paths, gan_paths) from a class folder."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    real, fake = [], []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        if p.name.lower().startswith("gan"):
            fake.append(p)
        else:
            real.append(p)
    return real, fake


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Training directory not found: {BASE_DIR.resolve()}")

    class_dirs = sorted([d for d in BASE_DIR.iterdir() if d.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No sub-folders found in {BASE_DIR}")

    print(f"Found {len(class_dirs)} class(es): {[d.name for d in class_dirs]}")
    print(f"Device: {DEVICE}\n")
    print("Loading Inception v3 …")
    model = build_inception()
    print("Model ready.\n")

    results = {}

    for cls_dir in class_dirs:
        cls_name = cls_dir.name
        real_paths, fake_paths = split_images(cls_dir)

        print(f"[{cls_name}]  real={len(real_paths)}  generated={len(fake_paths)}")

        if not real_paths:
            print("  ⚠ No real images found – skipping.\n")
            results[cls_name] = {"real": 0, "generated": 0, "fid": None,
                                  "note": "No real images"}
            continue
        if not fake_paths:
            print("  ⚠ No GAN images found – skipping.\n")
            results[cls_name] = {"real": len(real_paths), "generated": 0,
                                  "fid": None, "note": "No generated images"}
            continue

        print("  Extracting real features …")
        act_real = get_activations(real_paths,  model)
        print("  Extracting GAN features …")
        act_fake = get_activations(fake_paths, model)

        fid = compute_fid(act_real, act_fake)
        results[cls_name] = {
            "real":      len(real_paths),
            "generated": len(fake_paths),
            "fid":       round(fid, 4),
            "note":      "OK" if np.isfinite(fid) else "Too few samples for reliable FID",
        }
        print(f"  FID = {fid:.4f}\n")

    # ── Print summary ─────────────────────────────────────────────────────────
    valid_fids = [v["fid"] for v in results.values()
                  if v["fid"] is not None and np.isfinite(v["fid"])]

    print("=" * 60)
    print("  FID REPORT")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Base path: {BASE_DIR.resolve()}")
    print("=" * 60)
    print(f"  {'Class':<25}  {'Real':>6}  {'GAN':>6}  {'FID':>10}  Note")
    print("  " + "-" * 56)
    for cls, v in results.items():
        fid_str = f"{v['fid']:10.4f}" if v["fid"] is not None and np.isfinite(v["fid"]) else \
                  ("         inf" if v["fid"] == float("inf") else "          —")
        print(f"  {cls:<25}  {v['real']:>6}  {v['generated']:>6}  {fid_str}  {v['note']}")
    print("  " + "-" * 56)
    if valid_fids:
        print(f"  {'Overall average FID':<25}  {'':>6}  {'':>6}  {np.mean(valid_fids):>10.4f}")
    print("=" * 60)

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "generated_at": datetime.now().isoformat(),
        "base_path":    str(BASE_DIR.resolve()),
        "device":       str(DEVICE),
        "classes":      results,
        "average_fid":  round(float(np.mean(valid_fids)), 4) if valid_fids else None,
    }
    out_path = Path("fid_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved → {out_path.resolve()}")


if __name__ == "__main__":
    main()