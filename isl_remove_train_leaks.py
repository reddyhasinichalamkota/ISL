"""
ISL Static Sign Recognition - Remove Leaked Images from Training
Finds images that exist in BOTH Training and (Testing or Validation),
and deletes the copy in Training only. Testing/Validation are left
untouched since they're smaller and act as your "ground truth" sets.

Run isl_leakage_check.py FIRST to see the leaks before deleting.

HOW TO USE:
1. BASE_DIR is already set to your dataset path
2. Run: python isl_remove_train_leaks.py
3. It will print how many files it deletes from Training
"""

import hashlib
from pathlib import Path
from collections import defaultdict

BASE_DIR = r"C:\Users\reddy\OneDrive\Desktop\ISL_Translator\Dataset"
SPLITS = ["Training", "Testing", "Validation"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def hash_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def collect_hashes(base: Path):
    hash_map = defaultdict(list)
    for split in SPLITS:
        split_dir = base / split
        if not split_dir.exists():
            continue
        for class_folder in sorted(split_dir.iterdir()):
            if not class_folder.is_dir():
                continue
            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() not in IMG_EXTENSIONS:
                    continue
                try:
                    h = hash_file(img_path)
                    hash_map[h].append((split, img_path))
                except OSError:
                    continue
    return hash_map


def main():
    base = Path(BASE_DIR)
    print("Hashing all images...")
    hash_map = collect_hashes(base)

    deleted = 0
    for h, locs in hash_map.items():
        splits_present = {loc[0] for loc in locs}
        if len(splits_present) > 1 and "Training" in splits_present:
            # Delete only the Training copy/copies, keep Testing/Validation
            for split, path in locs:
                if split == "Training":
                    try:
                        path.unlink()
                        deleted += 1
                    except OSError as e:
                        print(f"  Could not delete {path}: {e}")

    print(f"\nDeleted {deleted} leaked images from Training.")
    print("Testing and Validation were left untouched.")
    print("Re-run isl_dataset_analysis.py to see updated Training counts.")


if __name__ == "__main__":
    main()