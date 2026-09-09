"""
ISL Static Sign Recognition - Cross-Split Leakage Check
Checks whether the SAME image appears in more than one split
(Training / Testing / Validation). This matters much more than
within-split duplicates, because if a training image leaks into
your test set, your accuracy numbers will be misleadingly high.

HOW TO USE:
1. Set BASE_DIR to your Dataset folder (same as before)
2. Run: python isl_leakage_check.py
"""

import hashlib
from pathlib import Path
from collections import defaultdict

# ------------------------------------------------------------------
# CONFIG - EDIT THIS PATH (same folder as your other script)
# ------------------------------------------------------------------
BASE_DIR = r"C:\Users\reddy\OneDrive\Desktop\ISL_Translator\Dataset"
SPLITS = ["Training", "Testing", "Validation"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def hash_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def collect_hashes(base: Path):
    """Return {hash: [(split, class, path), ...]} across all splits."""
    hash_map = defaultdict(list)

    for split in SPLITS:
        split_dir = base / split
        if not split_dir.exists():
            print(f"WARNING: {split_dir} not found, skipping.")
            continue

        for class_folder in sorted(split_dir.iterdir()):
            if not class_folder.is_dir():
                continue
            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() not in IMG_EXTENSIONS:
                    continue
                try:
                    h = hash_file(img_path)
                    hash_map[h].append((split, class_folder.name, str(img_path)))
                except OSError:
                    continue

    return hash_map


def main():
    print("=" * 60)
    print("CROSS-SPLIT LEAKAGE CHECK")
    print("=" * 60)

    base = Path(BASE_DIR)
    if not base.exists():
        print(f"ERROR: BASE_DIR not found: {BASE_DIR}")
        return

    print("Hashing all images across Training / Testing / Validation...")
    hash_map = collect_hashes(base)

    leaks = {h: locs for h, locs in hash_map.items() if len({loc[0] for loc in locs}) > 1}

    print(f"\nTotal unique images hashed: {len(hash_map)}")
    print(f"Images appearing in MORE THAN ONE split: {len(leaks)}")

    if leaks:
        print("\n--- Leakage examples (showing up to 10) ---")
        for i, (h, locs) in enumerate(leaks.items()):
            if i >= 10:
                break
            print(f"  Hash {h[:8]}... found in:")
            for split, cls, path in locs:
                print(f"    [{split}] {cls}: {path}")
    else:
        print("\nNo cross-split leakage found. Training, Testing, and "
              "Validation sets do not share any identical images.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()