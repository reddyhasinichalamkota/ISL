
"""
ISL Static Sign Recognition - Member 1, Review 1
Dataset analysis + preprocessing prototype

Covers Review 1 checkpoints:
- Dataset statistics (class counts, image counts)
- Corrupted / duplicate image check
- Class distribution plot
- Sample image display
- Resize + normalize preprocessing prototype

HOW TO USE:
1. Set BASE_DIR below to the folder that contains Training / Testing / Validation / Letters
2. Run: python isl_dataset_analysis.py
3. Outputs (plots, stats CSV) are saved into an "analysis_output" folder
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

# ------------------------------------------------------------------
# 1. CONFIG - EDIT THIS PATH
# ------------------------------------------------------------------
BASE_DIR = r"C:\Users\reddy\OneDrive\Desktop\ISL_Translator\Dataset"   # e.g. r"C:\Users\Hasi\Datasets\RealSign"
SPLITS = ["Training", "Testing", "Validation"]  # "Letters" is likely the master/reference folder, checked separately
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
TARGET_SIZE = (224, 224)  # MobileNetV2 default input size

OUTPUT_DIR = Path("analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------------
# 2. DATASET STATISTICS
# ------------------------------------------------------------------
def scan_split(split_dir: Path):
    """Return {class_name: [list of image paths]} for one split folder."""
    class_map = defaultdict(list)
    if not split_dir.exists():
        print(f"  WARNING: {split_dir} does not exist, skipping.")
        return class_map

    for class_folder in sorted(split_dir.iterdir()):
        if class_folder.is_dir():
            imgs = [
                p for p in class_folder.iterdir()
                if p.suffix.lower() in IMG_EXTENSIONS
            ]
            class_map[class_folder.name] = imgs
    return class_map


def print_stats(name, class_map):
    print(f"\n--- {name} ---")
    total = 0
    for cls, imgs in class_map.items():
        print(f"  {cls}: {len(imgs)} images")
        total += len(imgs)
    print(f"  TOTAL: {total} images across {len(class_map)} classes")
    return total


# ------------------------------------------------------------------
# 3. CORRUPTED / DUPLICATE CHECK
# ------------------------------------------------------------------
def check_corrupted(class_map):
    corrupted = []
    for cls, imgs in class_map.items():
        for img_path in imgs:
            try:
                with Image.open(img_path) as im:
                    im.verify()
            except (UnidentifiedImageError, OSError):
                corrupted.append(str(img_path))
    return corrupted


def check_duplicates(class_map):
    """Hash-based duplicate check within the same split."""
    hashes = defaultdict(list)
    for cls, imgs in class_map.items():
        for img_path in imgs:
            try:
                with open(img_path, "rb") as f:
                    h = hashlib.md5(f.read()).hexdigest()
                hashes[h].append(str(img_path))
            except OSError:
                continue
    duplicates = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    return duplicates


# ------------------------------------------------------------------
# 4. CLASS DISTRIBUTION PLOT
# ------------------------------------------------------------------
def plot_class_distribution(class_map, split_name):
    classes = sorted(class_map.keys())
    counts = [len(class_map[c]) for c in classes]

    plt.figure(figsize=(14, 5))
    plt.bar(classes, counts, color="#4C72B0")
    plt.xlabel("Class (Letter)")
    plt.ylabel("Number of Images")
    plt.title(f"Class Distribution - {split_name}")
    plt.xticks(rotation=90)
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"class_distribution_{split_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved distribution plot -> {out_path}")


# ------------------------------------------------------------------
# 5. SAMPLE IMAGE DISPLAY
# ------------------------------------------------------------------
def show_sample_images(class_map, split_name, n_classes=6):
    classes = sorted(class_map.keys())[:n_classes]
    fig, axes = plt.subplots(1, len(classes), figsize=(3 * len(classes), 3))
    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        imgs = class_map[cls]
        if imgs:
            img = Image.open(imgs[0])
            ax.imshow(img)
            ax.set_title(cls)
        ax.axis("off")

    plt.tight_layout()
    out_path = OUTPUT_DIR / f"sample_images_{split_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved sample images -> {out_path}")


# ------------------------------------------------------------------
# 6. PREPROCESSING PROTOTYPE (resize + normalize)
# ------------------------------------------------------------------
def preprocess_image(img_path, target_size=TARGET_SIZE):
    """Load, resize, and normalize a single image to [0, 1] float array."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    return arr


def preprocessing_prototype_demo(class_map):
    print("\n--- Preprocessing Prototype Demo ---")
    first_class = sorted(class_map.keys())[0]
    sample_path = class_map[first_class][0]
    arr = preprocess_image(sample_path)
    print(f"  Sample image: {sample_path}")
    print(f"  Output shape: {arr.shape}, dtype: {arr.dtype}, "
          f"min: {arr.min():.3f}, max: {arr.max():.3f}")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("ISL STATIC DATASET ANALYSIS - Member 1, Review 1")
    print("=" * 60)

    base = Path(BASE_DIR)
    if not base.exists():
        print(f"\nERROR: BASE_DIR does not exist: {BASE_DIR}")
        print("Edit BASE_DIR at the top of this script and re-run.")
        return

    all_stats = {}

    for split in SPLITS:
        split_dir = base / split
        class_map = scan_split(split_dir)
        if not class_map:
            continue

        total = print_stats(split, class_map)
        all_stats[split] = total

        corrupted = check_corrupted(class_map)
        print(f"  Corrupted images found: {len(corrupted)}")
        if corrupted:
            print(f"    e.g. {corrupted[:3]}")

        duplicates = check_duplicates(class_map)
        print(f"  Duplicate groups found: {len(duplicates)}")

        plot_class_distribution(class_map, split)
        show_sample_images(class_map, split)

        if split == "Training":
            preprocessing_prototype_demo(class_map)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for split, total in all_stats.items():
        print(f"  {split}: {total} images")
    print(f"\nAll plots saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()