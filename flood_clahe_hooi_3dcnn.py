#!/usr/bin/env python3
"""
Sentinel-2 flood / non-flood patch classification
CLAHE -> HOOI (Tucker decomposition) -> CLAHE -> 3D-CNN



Important methodological alignment:
- Input imagery: Sentinel-2 bands B12, B11 and B2, in that order.
- Preprocessing: per-band min-max normalization followed by per-scene
  +/- 2 standard-deviation clipping.
- Tensor compression: Tucker decomposition optimized by HOOI.
- Contrast enhancement: CLAHE.
- Patch extraction: 32 x 32 patches.
- Classifier: 3D-CNN with a single-channel, depth-one input
  (N, 1, 1, 32, 32), matching the manuscript's explicit implementation
  description. With depth=1, the network is technically a 3D convolutional
  implementation operating on a 2D grayscale patch; it does NOT learn
  cross-band spectral depth at the classifier stage.
- Split: 70% train, 15% validation, 15% test.
- Training: Adam, lr=1e-4, batch size=32, up to 25 epochs, early stopping.

Dataset assumptions:
A. The simplest supported layout is:
       DATASET/
          flood/       *.tif, *.tiff
          no_flood/    *.tif, *.tiff
   Folder names are configurable below.
B. Alternatively, class labels can be inferred from filenames containing
   configurable flood/non-flood keywords.
C. Each image should be a multi-band GeoTIFF containing B12, B11 and B2.
   Set BAND_INDEXES below to the zero-based positions of those bands in
   your GeoTIFF. The defaults assume a 12-band stack ordered B1...B12:
       B12 -> index 11, B11 -> index 10, B2 -> index 1.



Install dependencies:
    pip install numpy matplotlib scikit-learn scikit-image tifffile rasterio torch tensorly

Run:
    python flood_clahe_hooi_3dcnn.py --data_dir "C:/path/to/Flood_dataset"

Optional:
    python flood_clahe_hooi_3dcnn.py --data_dir DATASET --output_dir results
"""

from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from skimage import exposure
import tensorly as tl
from tensorly.decomposition import tucker


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

# Zero-based indexes in a 12-band GeoTIFF stack ordered B1, B2, ..., B12.

BAND_INDEXES = (11, 10, 1)

FLOOD_DIR_NAMES = ("flood", "flooded", "with_flood")
NON_FLOOD_DIR_NAMES = ("no_flood", "non_flood", "non-flood", "nonflood", "no-flood")
FLOOD_FILENAME_KEYWORDS = ("flood", "flooded")
NON_FLOOD_FILENAME_KEYWORDS = ("no_flood", "non_flood", "non-flood",
                               "nonflood", "no-flood", "noflood")

PATCH_SIZE = 32
PATCH_STRIDE = 32  # manuscript specifies 32x32 patches; use 32 for non-overlap
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MAX_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 5
RANDOM_SEED = 42

# HOOI/Tucker rank configuration.


HOOI_SPATIAL_RANK = None  # None = retain full H and W dimensions
HOOI_SPECTRAL_RANK = 2
HOOI_N_ITERATIONS = 10

CLAHE_CLIP_LIMIT = 0.03
CLAHE_KERNEL_SIZE = (8, 8)

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Reproducibility and I/O
# ---------------------------------------------------------------------------

def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_geotiff(path: Path) -> np.ndarray:
    """Read a GeoTIFF as H x W x bands, handling common tifffile layouts."""
    arr = tifffile.imread(str(path))
    arr = np.asarray(arr)

    if arr.ndim == 2:
        raise ValueError(
            f"{path.name} is single-band. The manuscript workflow requires "
            "the B12/B11/B2 composite; provide a multi-band stack or adapt "
            "load_geotiff() to read separate band files."
        )

    if arr.ndim != 3:
        raise ValueError(f"Unsupported TIFF shape {arr.shape} in {path.name}")

    # Convert common band-first layouts (bands, H, W) to (H, W, bands).
    # Heuristic: a small first dimension is likely the band dimension.
    if arr.shape[0] <= 20 and arr.shape[-1] > 20:
        arr = np.moveaxis(arr, 0, -1)

    if arr.shape[-1] <= max(BAND_INDEXES):
        raise ValueError(
            f"{path.name} has {arr.shape[-1]} bands, but BAND_INDEXES="
            f"{BAND_INDEXES}. Check the band ordering/indexes."
        )

    return arr.astype(np.float32)


def infer_label(path: Path, root: Path) -> Optional[int]:
    """Return 1=flood, 0=non-flood, or None if the label is ambiguous."""
    relative_parts = [part.lower() for part in path.relative_to(root).parts[:-1]]

    for part in relative_parts:
        if part in FLOOD_DIR_NAMES:
            return 1
        if part in NON_FLOOD_DIR_NAMES:
            return 0

    name = path.stem.lower()
    is_non_flood = any(k in name for k in NON_FLOOD_FILENAME_KEYWORDS)
    is_flood = any(k in name for k in FLOOD_FILENAME_KEYWORDS)

    # Check negative labels first because "no_flood" contains "flood".
    if is_non_flood:
        return 0
    if is_flood:
        return 1
    return None


# ---------------------------------------------------------------------------
# Preprocessing: B12/B11/B2 -> normalization -> clipping -> HOOI -> CLAHE
# ---------------------------------------------------------------------------

def select_manuscript_bands(image: np.ndarray) -> np.ndarray:
    """Select B12, B11, B2 in the manuscript's stated order."""
    return image[..., list(BAND_INDEXES)]


def normalize_per_band(image: np.ndarray) -> np.ndarray:
    """Per-scene, per-band min-max normalization to [0, 1]."""
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros_like(image, dtype=np.float32)
    for band_idx in range(image.shape[-1]):
        band = image[..., band_idx]
        lo = float(np.min(band))
        hi = float(np.max(band))
        if hi > lo:
            out[..., band_idx] = (band - lo) / (hi - lo)
        else:
            out[..., band_idx] = 0.0
    return out


def clip_outliers(image: np.ndarray, k: float = 2.0) -> np.ndarray:
    """
    Clip each band's pixels to mean +/- k*std, calculated per scene.
    This is statistical clipping; it is not a dedicated cloud mask.
    """
    out = np.empty_like(image, dtype=np.float32)
    for band_idx in range(image.shape[-1]):
        band = image[..., band_idx]
        mean = float(np.mean(band))
        std = float(np.std(band))
        if std == 0:
            out[..., band_idx] = band
        else:
            out[..., band_idx] = np.clip(band, mean - k * std, mean + k * std)
    return np.clip(out, 0.0, 1.0)


def hooi_tucker_reconstruct(image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Apply Tucker decomposition using TensorLy's HOOI implementation, then
    reconstruct the image tensor as described in the manuscript.

    Note: reconstructing to the original dimensions preserves a convenient
    image for CLAHE, but the full compression benefit is not retained in the
    reconstructed output. The low-rank core is still used to approximate it.
    """
    h, w, bands = image.shape
    rank_h = h if HOOI_SPATIAL_RANK is None else min(HOOI_SPATIAL_RANK, h)
    rank_w = w if HOOI_SPATIAL_RANK is None else min(HOOI_SPATIAL_RANK, w)
    rank_b = max(1, min(HOOI_SPECTRAL_RANK, bands))
    ranks = (rank_h, rank_w, rank_b)

    tensor = tl.tensor(image, dtype=tl.float32)
    try:
        core, factors = tucker(
            tensor,
            rank=ranks,
            n_iter_max=HOOI_N_ITERATIONS,
            init="svd",
            tol=1e-5,
        )
    except TypeError:
        # Compatibility with TensorLy versions that do not expose method.
        core, factors = tucker(
            tensor,
            rank=ranks,
            n_iter_max=HOOI_N_ITERATIONS,
            init="svd",
            tol=1e-5,
        )

    reconstructed = tl.tucker_to_tensor((core, factors))
    reconstructed = np.asarray(reconstructed, dtype=np.float32)
    reconstructed = np.clip(reconstructed, 0.0, 1.0)

    info = {
        "input_shape": [int(x) for x in image.shape],
        "core_shape": [int(x) for x in core.shape],
        "tucker_ranks": [int(x) for x in ranks],
        "iterations": HOOI_N_ITERATIONS,
    }
    return reconstructed, info


def apply_clahe_to_composite(image: np.ndarray) -> np.ndarray:
    """
    Convert the reconstructed 3-band B12/B11/B2 composite to one grayscale
    channel, then apply CLAHE. 
    """
  
    gray = np.mean(image, axis=-1)
    gray = np.clip(gray, 0.0, 1.0)

    enhanced = exposure.equalize_adapthist(
        gray,
        kernel_size=CLAHE_KERNEL_SIZE,
        clip_limit=CLAHE_CLIP_LIMIT,
    )
    return np.asarray(enhanced, dtype=np.float32)


def preprocess_scene(raw_image: np.ndarray) -> Tuple[np.ndarray, Dict]:
    selected = select_manuscript_bands(raw_image)
    normalized = normalize_per_band(selected)
    clipped = clip_outliers(normalized, k=2.0)
    reconstructed, hooi_info = hooi_tucker_reconstruct(clipped)
    enhanced_gray = apply_clahe_to_composite(reconstructed)
    return enhanced_gray, hooi_info


# ---------------------------------------------------------------------------
# Patch extraction and labelled dataset construction
# ---------------------------------------------------------------------------

def extract_patches(image: np.ndarray, label: int, patch_size: int = PATCH_SIZE,
                    stride: int = PATCH_STRIDE) -> Tuple[List[np.ndarray], List[int]]:
    """Extract 32x32 grayscale patches; each inherits its source-image label."""
    h, w = image.shape
    patches, labels = [], []

    if h < patch_size or w < patch_size:
        warnings.warn(f"Skipping image smaller than {patch_size}x{patch_size}: {image.shape}")
        return patches, labels

    for top in range(0, h - patch_size + 1, stride):
        for left in range(0, w - patch_size + 1, stride):
            patch = image[top:top + patch_size, left:left + patch_size]
            if patch.shape == (patch_size, patch_size):
                patches.append(patch.astype(np.float32))
                labels.append(int(label))
    return patches, labels


def build_patch_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    files = sorted(
        list(data_dir.rglob("*.tif")) + list(data_dir.rglob("*.tiff"))
    )
    if not files:
        raise FileNotFoundError(f"No .tif/.tiff files found under {data_dir}")

    all_patches: List[np.ndarray] = []
    all_labels: List[int] = []
    scene_log: List[Dict] = []
    skipped = []

    for file_path in files:
        label = infer_label(file_path, data_dir)
        if label is None:
            skipped.append(str(file_path))
            continue

        try:
            raw = load_geotiff(file_path)
            processed, hooi_info = preprocess_scene(raw)
            patches, labels = extract_patches(processed, label)
            all_patches.extend(patches)
            all_labels.extend(labels)
            scene_log.append({
                "file": str(file_path),
                "label": "flood" if label == 1 else "non_flood",
                "raw_shape": [int(x) for x in raw.shape],
                "processed_shape": [int(x) for x in processed.shape],
                "patch_count": len(patches),
                "hooi": hooi_info,
            })
            print(f"[OK] {file_path.name}: {len(patches)} patches "
                  f"({'flood' if label else 'non-flood'})")
        except Exception as exc:
            warnings.warn(f"Could not process {file_path}: {exc}")

    if skipped:
        print("\nFiles skipped because no class label could be inferred:")
        for item in skipped:
            print("  ", item)

    if not all_patches:
        raise RuntimeError(
            "No labelled patches were generated. Check folder names, filenames, "
            "GeoTIFF band count/order, and image dimensions."
        )

    X = np.stack(all_patches).astype(np.float32)
    y = np.asarray(all_labels, dtype=np.int64)

    unique, counts = np.unique(y, return_counts=True)
    distribution = {("non_flood" if int(k) == 0 else "flood"): int(v)
                    for k, v in zip(unique, counts)}
    print(f"\nTotal patches: {len(y)}; class distribution: {distribution}")
    return X, y, scene_log


# ---------------------------------------------------------------------------
# Dataset and model
# ---------------------------------------------------------------------------

class FloodPatchDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        # (N,H,W) -> (N, C=1, D=1, H, W)
        self.images = torch.from_numpy(images[:, None, None, :, :]).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


class SpatialAttention3D(nn.Module):
    """Compact 3D spatial attention block returning the same feature shape."""
    def __init__(self, channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class Manuscript3DCNN(nn.Module):
    """
    Layer sequence based on manuscript Table 2:
    Conv3D(16) -> Conv3D(32) -> MaxPool -> Conv3D(64)
    -> 3D spatial attention -> Global Average Pool -> FC(2).
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SpatialAttention3D(64),
        )
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Training, validation, testing, metrics and plots
# ---------------------------------------------------------------------------

def make_loaders(X: np.ndarray, y: np.ndarray):
    if len(np.unique(y)) < 2:
        raise ValueError("Both flood and non-flood classes are required.")

    # First reserve 15% test; then reserve 15% of the full dataset for val.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_SEED, stratify=y
    )
    val_fraction_of_trainval = VALIDATION_RATIO / (TRAIN_RATIO + VALIDATION_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction_of_trainval,
        random_state=RANDOM_SEED,
        stratify=y_trainval,
    )

    loaders = {
        "train": DataLoader(FloodPatchDataset(X_train, y_train),
                            batch_size=BATCH_SIZE, shuffle=True),
        "val": DataLoader(FloodPatchDataset(X_val, y_val),
                          batch_size=BATCH_SIZE, shuffle=False),
        "test": DataLoader(FloodPatchDataset(X_test, y_test),
                           batch_size=BATCH_SIZE, shuffle=False),
    }
    print(f"Split sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    return loaders


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
    return total_loss / max(total, 1)


def train_model(model, loaders, output_dir: Path):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(MAX_EPOCHS):
        model.train()
        running_loss, seen = 0.0, 0

        for images, labels in loaders["train"]:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        train_loss = running_loss / max(seen, 1)
        val_loss = evaluate_loss(model, loaders["val"], criterion, DEVICE)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch + 1:02d}/{MAX_EPOCHS} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), output_dir / "flood_3dcnn_best.pth")
    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return model, history


def predict(model, loader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    start = time.perf_counter()
    n_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
            n_samples += labels.size(0)

    elapsed = time.perf_counter() - start
    latency_ms = (elapsed * 1000.0 / max(n_samples, 1))
    return np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob), latency_ms


def calculate_metrics(y_true, y_pred, y_prob) -> Dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["non_flood", "flood"],
            zero_division=0, output_dict=True
        ),
    }
    metrics["auc"] = (
        float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) == 2 else None
    )
    return metrics


def save_plots(history, y_true, y_prob, output_dir: Path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    plt.figure(figsize=(7, 5))
    plt.plot(history["train_loss"], label="Training loss")
    plt.plot(history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300)
    plt.close()

    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"ROC (AUC={auc_value:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("Test-set ROC curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / "test_roc_curve.png", dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Sentinel-2 flood classification using CLAHE, HOOI and 3D-CNN"
    )
    parser.add_argument("--data_dir", required=True,
                        help="Root folder containing labelled flood/non-flood GeoTIFFs")
    parser.add_argument("--output_dir", default="flood_results",
                        help="Folder for model, metrics, logs and plots")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {data_dir}")

    set_seed()
    print(f"Device: {DEVICE}")
    print(f"Dataset: {data_dir}")
    print(f"Selected band indexes (B12, B11, B2): {BAND_INDEXES}")

    X, y, scene_log = build_patch_dataset(data_dir)
    loaders = make_loaders(X, y)

    model = Manuscript3DCNN(num_classes=2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")
    model, history = train_model(model, loaders, output_dir)

    y_true, y_pred, y_prob, latency_ms = predict(model, loaders["test"])
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    metrics["average_test_inference_ms_per_patch"] = float(latency_ms)
    metrics["trainable_parameters"] = int(n_params)
    metrics["device"] = str(DEVICE)

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / "scene_processing_log.json", "w", encoding="utf-8") as f:
        json.dump(scene_log, f, indent=2)

    config = {
        "band_indexes_zero_based": BAND_INDEXES,
        "band_order": ["B12", "B11", "B2"],
        "normalization": "per-scene, per-band min-max",
        "outlier_clipping": "per-scene, per-band mean +/- 2 standard deviations",
        "hooi": {
            "method": "Tucker decomposition optimized by HOOI",
            "spatial_rank": HOOI_SPATIAL_RANK,
            "spectral_rank": HOOI_SPECTRAL_RANK,
            "iterations": HOOI_N_ITERATIONS,
        },
        "clahe": {
            "clip_limit": CLAHE_CLIP_LIMIT,
            "kernel_size": CLAHE_KERNEL_SIZE,
            "grayscale_conversion": "mean of reconstructed B12/B11/B2 channels",
        },
        "patch_size": PATCH_SIZE,
        "patch_stride": PATCH_STRIDE,
        "split": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO,
        },
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "seed": RANDOM_SEED,
        "model_input_shape": "(N, 1, 1, 32, 32)",
    }
    with open(output_dir / "run_configuration.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    save_plots(history, y_true, y_prob, output_dir)

    print("\nTEST RESULTS")
    print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall   : {metrics['recall'] * 100:.2f}%")
    print(f"F1-score : {metrics['f1_score'] * 100:.2f}%")
    print(f"AUC      : {metrics['auc']}")
    print(f"Latency  : {latency_ms:.2f} ms/patch (test loader average)")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
