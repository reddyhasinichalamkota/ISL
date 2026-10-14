"""
Unit tests for src.dataset package:
- Dataset loader
- Integrity validator
- Stratified splitter
"""

import os
import unittest
import tempfile
import pandas as pd
import numpy as np
from PIL import Image

from src.dataset.split_dataset import split_dataframe_stratified
from src.dataset.dataset_validator import hash_file, find_corrupted_images, find_duplicate_images
from src.dataset.dataset_loader import load_static_tensorflow_datasets


class TestDatasetPackage(unittest.TestCase):

    def test_stratified_split(self):
        # Create a synthetic dataset with 100 rows across 5 classes
        data = {
            "video_path": [f"video_{i}.mp4" for i in range(100)],
            "class_label": [f"Class_{i % 5}" for i in range(100)]
        }
        df = pd.DataFrame(data)

        train_df, val_df, test_df = split_dataframe_stratified(
            df,
            label_column="class_label",
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            random_state=42
        )

        self.assertEqual(len(train_df), 70)
        self.assertEqual(len(val_df), 15)
        self.assertEqual(len(test_df), 15)
        self.assertEqual(set(train_df["split"]), {"train"})
        self.assertEqual(set(val_df["split"]), {"val"})
        self.assertEqual(set(test_df["split"]), {"test"})

    def test_dataset_validator_integrity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid image
            img_path = os.path.join(temp_dir, "valid.jpg")
            img = Image.new("RGB", (32, 32), color="red")
            img.save(img_path)

            # Create an identical duplicate
            dup_path = os.path.join(temp_dir, "dup.jpg")
            img.save(dup_path)

            # Create a corrupt image file
            bad_path = os.path.join(temp_dir, "bad.jpg")
            with open(bad_path, "wb") as f:
                f.write(b"not an image data")

            # Check hashing
            h1 = hash_file(img_path)
            h2 = hash_file(dup_path)
            self.assertEqual(h1, h2)

            # Check duplicate detection
            dups = find_duplicate_images(temp_dir)
            self.assertIn(h1, dups)
            self.assertEqual(len(dups[h1]), 2)

            # Check corrupt detection
            corrupt = find_corrupted_images(temp_dir)
            self.assertEqual(len(corrupt), 1)
            self.assertTrue(corrupt[0][0].endswith("bad.jpg"))

    def test_dataset_loader_missing_dir_error(self):
        with self.assertRaises(FileNotFoundError):
            load_static_tensorflow_datasets("non_existent_directory_12345")


if __name__ == "__main__":
    unittest.main()
