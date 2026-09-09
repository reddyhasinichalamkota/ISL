
# =========================================================
# NOTEBOOK CELL 0
# =========================================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

EXCLUDED_FOLDERS = {"Still"}  # still-sign videos, not part of dynamic pipeline
dataset_path="Video_Dataset"

def collect_videos(dataset_path):
    records = []
    class_folders = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d)) and d not in EXCLUDED_FOLDERS
    ])
    print(f"Found {len(class_folders)} class folders (after excluding {EXCLUDED_FOLDERS}).")

    for class_name in class_folders:
        class_dir = os.path.join(dataset_path, class_name)
        video_files = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(".mp4")])
        for vf in video_files:
            records.append({"video_path": os.path.join(class_dir, vf), "class_label": class_name})

    return pd.DataFrame(records), class_folders

df, class_folders = collect_videos(dataset_path)

print(f"\nTotal videos collected: {len(df)}")
print(f"Total classes: {df['class_label'].nunique()}")

videos_per_class = df.groupby("class_label").size()
print("Videos per class (min/max/mean):", videos_per_class.min(), videos_per_class.max(), videos_per_class.mean())

if len(df) != 3600:
    print(f"WARNING: Expected 3600 videos, found {len(df)}.")
else:
    print("Video count matches expected 3600.")

# Stratified 70/15/15 split
train_df, temp_df = train_test_split(
    df, test_size=(VAL_RATIO + TEST_RATIO),
    stratify=df["class_label"], random_state=RANDOM_SEED
)
relative_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
val_df, test_df = train_test_split(
    temp_df, test_size=relative_test_ratio,
    stratify=temp_df["class_label"], random_state=RANDOM_SEED
)

train_df = train_df.copy(); val_df = val_df.copy(); test_df = test_df.copy()
train_df["split"] = "train"; val_df["split"] = "val"; test_df["split"] = "test"

full_df = pd.concat([train_df, val_df, test_df], axis=0).reset_index(drop=True)
output_csv_path = "dynamic_video_splits.csv"  
full_df.to_csv(output_csv_path, index=False)
print(f"\nSaved split CSV to: {output_csv_path}")

print("\n=== Overall split sizes ===")
print(full_df["split"].value_counts())
print("\n=== Split proportions ===")
print((full_df["split"].value_counts(normalize=True) * 100).round(2))

split_per_class = full_df.groupby(["class_label", "split"]).size().unstack(fill_value=0)
print("\n=== Min/Max videos per class per split ===")
print(split_per_class.min())
print(split_per_class.max())
print("\n=== Total videos in final CSV ===", len(full_df))


# =========================================================
# NOTEBOOK CELL 1
# =========================================================

import pandas as pd

# ---------------------------------------------------------
# LOAD SPLIT CSV
# ---------------------------------------------------------
csv_path = "dynamic_video_splits.csv"   
df = pd.read_csv(csv_path)

print(f"Loaded CSV: {csv_path}")
print(f"Total rows: {len(df)}")

# ---------------------------------------------------------
# STEP 2: LEAKAGE CHECK
# ---------------------------------------------------------

# Get video_path sets per split
train_paths = set(df.loc[df["split"] == "train", "video_path"])
val_paths   = set(df.loc[df["split"] == "val",   "video_path"])
test_paths  = set(df.loc[df["split"] == "test",  "video_path"])

print(f"\nTrain videos: {len(train_paths)}")
print(f"Val videos:   {len(val_paths)}")
print(f"Test videos:  {len(test_paths)}")

# ---------------------------------------------------------
# Check 1 & 6: pairwise overlaps
# ---------------------------------------------------------
overlap_train_val  = train_paths & val_paths
overlap_train_test = train_paths & test_paths
overlap_val_test    = val_paths & test_paths

print("\n=== Pairwise Overlap Counts ===")
print(f"Train ∩ Val:   {len(overlap_train_val)}")
print(f"Train ∩ Test:  {len(overlap_train_test)}")
print(f"Val ∩ Test:    {len(overlap_val_test)}")

if overlap_train_val:
    print("Sample overlapping (train/val):", list(overlap_train_val)[:5])
if overlap_train_test:
    print("Sample overlapping (train/test):", list(overlap_train_test)[:5])
if overlap_val_test:
    print("Sample overlapping (val/test):", list(overlap_val_test)[:5])

# ---------------------------------------------------------
# Check 5: unique paths vs total rows (catches duplicate rows too,
# e.g. same video appearing twice within the SAME split)
# ---------------------------------------------------------
total_rows = len(df)
unique_paths = df["video_path"].nunique()

print(f"\nTotal rows in CSV: {total_rows}")
print(f"Unique video_paths in CSV: {unique_paths}")

duplicates_within_csv = total_rows - unique_paths
if duplicates_within_csv > 0:
    dup_rows = df[df.duplicated(subset="video_path", keep=False)].sort_values("video_path")
    print(f"WARNING: {duplicates_within_csv} duplicate video_path row(s) found (may or may not cross splits).")
    print(dup_rows.head(10))

# ---------------------------------------------------------
# FINAL PASS/FAIL
# ---------------------------------------------------------
no_pairwise_leakage = (
    len(overlap_train_val) == 0 and
    len(overlap_train_test) == 0 and
    len(overlap_val_test) == 0
)
no_duplicate_rows = (duplicates_within_csv == 0)

print("\n================ LEAKAGE CHECK RESULT ================")
print(f"1. No overlap Train/Val:        {'PASS' if len(overlap_train_val) == 0 else 'FAIL'}")
print(f"2. No overlap Train/Test:       {'PASS' if len(overlap_train_test) == 0 else 'FAIL'}")
print(f"3. No overlap Val/Test:         {'PASS' if len(overlap_val_test) == 0 else 'FAIL'}")
print(f"4. Unique paths == total rows:  {'PASS' if no_duplicate_rows else 'FAIL'} "
      f"({unique_paths} unique / {total_rows} total)")

overall_result = no_pairwise_leakage and no_duplicate_rows
print("\n=========================================================")
print(f"OVERALL LEAKAGE CHECK: {'PASS ✅' if overall_result else 'FAIL ❌'}")
print("=========================================================")


# =========================================================
# NOTEBOOK CELL 2
# =========================================================

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
csv_path = "dynamic_video_splits.csv"
df = pd.read_csv(csv_path)

# ---------------------------------------------------------
# LABEL ENCODING (fixed, reproducible mapping)
# ---------------------------------------------------------
class_names = sorted(df["class_label"].unique())
label_to_idx = {name: idx for idx, name in enumerate(class_names)}
idx_to_label = {idx: name for name, idx in label_to_idx.items()}

print(f"Total classes: {len(class_names)}")
print("First 5 label mappings:", list(label_to_idx.items())[:5])

# ---------------------------------------------------------
# STEP 3: RAW VIDEO DATASET (no fixed-length sampling / preprocessing yet)
# ---------------------------------------------------------
class SignLanguageVideoDataset(Dataset):
    """
    Loads RAW frames (all frames, variable length) from each video.
    No resizing, normalization, or fixed sequence sampling here —
    those are handled in later steps.
    """
    def __init__(self, dataframe, split, label_to_idx):
        self.df = dataframe[dataframe["split"] == split].reset_index(drop=True)
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR -> RGB for consistency with standard CNN preprocessing later
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames read from video: {video_path}")

        return np.stack(frames, axis=0)  # shape: (T, H, W, C)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        class_label = row["class_label"]

        frames = self._load_video_frames(video_path)  # (T, H, W, C), raw uint8
        label_idx = self.label_to_idx[class_label]

        sample = {
            "frames": frames,             # numpy array, variable T
            "label": label_idx,
            "class_name": class_label,
            "video_path": video_path,
            "num_frames": frames.shape[0]
        }
        return sample

# ---------------------------------------------------------
# INSTANTIATE DATASETS PER SPLIT
# ---------------------------------------------------------
train_dataset = SignLanguageVideoDataset(df, split="train", label_to_idx=label_to_idx)
val_dataset   = SignLanguageVideoDataset(df, split="val",   label_to_idx=label_to_idx)
test_dataset  = SignLanguageVideoDataset(df, split="test",  label_to_idx=label_to_idx)

print(f"\nTrain dataset size: {len(train_dataset)}")
print(f"Val dataset size:   {len(val_dataset)}")
print(f"Test dataset size:  {len(test_dataset)}")

# ---------------------------------------------------------
# TEST: FETCH ONE SAMPLE DIRECTLY (no DataLoader)
# ---------------------------------------------------------
sample = train_dataset[0]
print("\n=== Single sample check (train_dataset[0]) ===")
print("video_path:", sample["video_path"])
print("class_name:", sample["class_name"])
print("label (idx):", sample["label"])
print("frames shape (T, H, W, C):", sample["frames"].shape)
print("frames dtype:", sample["frames"].dtype)
print("num_frames:", sample["num_frames"])

# ---------------------------------------------------------
# CUSTOM COLLATE FN (needed since raw frame counts differ per video)
# For this step, just pass samples through as a list — no stacking yet.
# ---------------------------------------------------------
def raw_collate_fn(batch):
    # batch is a list of sample dicts; can't torch.stack frames (variable T)
    return batch

# ---------------------------------------------------------
# DATALOADER TEST (batch_size=1 only — variable-length frames block batching >1)
# ---------------------------------------------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    collate_fn=raw_collate_fn
)

print("\n=== DataLoader smoke test (3 batches, batch_size=1) ===")
for i, batch in enumerate(train_loader):
    item = batch[0]
    print(f"Batch {i}: video={os.path.basename(item['video_path'])}, "
          f"class={item['class_name']}, frames={item['frames'].shape}")
    if i == 2:
        break

print("\nStep 3 complete: Dataset and DataLoader (batch_size=1) verified.")
print("Note: full batching will be enabled in Step 4 after fixed-length sequence sampling.")


# =========================================================
# NOTEBOOK CELL 3
# =========================================================

import numpy as np

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
SEQUENCE_LENGTH = 90  # matches your validated test (118 -> 90) and dataset min (90)

# ---------------------------------------------------------
# UNIFORM TEMPORAL SAMPLING FUNCTION
# ---------------------------------------------------------
def uniform_temporal_sample(frames, target_len=SEQUENCE_LENGTH):
    """
    Uniformly samples `target_len` frames from a (T, H, W, C) array.
    Assumes T >= target_len (true for this dataset: min T = 90).
    Uses evenly spaced indices covering the full temporal span.
    """
    total_frames = frames.shape[0]

    if total_frames < target_len:
        # Safety fallback (shouldn't trigger given min=90), pad by repeating last frame
        pad_count = target_len - total_frames
        pad = np.repeat(frames[-1:], pad_count, axis=0)
        sampled = np.concatenate([frames, pad], axis=0)
        return sampled

    # Evenly spaced indices from 0 to total_frames-1, length = target_len
    indices = np.linspace(0, total_frames - 1, num=target_len)
    indices = np.round(indices).astype(int)
    indices = np.clip(indices, 0, total_frames - 1)

    sampled = frames[indices]
    return sampled


# ---------------------------------------------------------
# EXTEND DATASET CLASS: adds fixed-length sequence output
# ---------------------------------------------------------
class SignLanguageVideoDataset(Dataset):
    def __init__(self, dataframe, split, label_to_idx, sequence_length=SEQUENCE_LENGTH):
        self.df = dataframe[dataframe["split"] == split].reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.df)

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path,cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames read from video: {video_path}")

        return np.stack(frames, axis=0)  # (T, H, W, C)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        class_label = row["class_label"]

        raw_frames = self._load_video_frames(video_path)          # (T, H, W, C)
        sampled_frames = uniform_temporal_sample(raw_frames, self.sequence_length)  # (SEQ_LEN, H, W, C)

        label_idx = self.label_to_idx[class_label]

        sample = {
            "frames": sampled_frames,          # fixed length now
            "label": label_idx,
            "class_name": class_label,
            "video_path": video_path,
            "original_num_frames": raw_frames.shape[0],
            "sampled_num_frames": sampled_frames.shape[0]
        }
        return sample


# ---------------------------------------------------------
# REBUILD DATASETS WITH FIXED SEQUENCE SAMPLING
# ---------------------------------------------------------
train_dataset = SignLanguageVideoDataset(df, split="train", label_to_idx=label_to_idx)
val_dataset   = SignLanguageVideoDataset(df, split="val",   label_to_idx=label_to_idx)
test_dataset  = SignLanguageVideoDataset(df, split="test",  label_to_idx=label_to_idx)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size:   {len(val_dataset)}")
print(f"Test dataset size:  {len(test_dataset)}")

# ---------------------------------------------------------
# TEST: SINGLE SAMPLE — VERIFY FIXED LENGTH
# ---------------------------------------------------------
sample = train_dataset[0]
print("\n=== Single sample check (train_dataset[0]) ===")
print("video_path:", sample["video_path"])
print("class_name:", sample["class_name"])
print("original_num_frames:", sample["original_num_frames"])
print("sampled_num_frames:", sample["sampled_num_frames"])
print("frames shape (T, H, W, C):", sample["frames"].shape)
print("frames dtype:", sample["frames"].dtype)

# ---------------------------------------------------------
# TEST: MULTIPLE SAMPLES ACROSS DATASET — CONFIRM ALL == SEQUENCE_LENGTH
# ---------------------------------------------------------
def check_sequence_lengths(dataset, n=50):
    lengths = []
    for i in np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False):
        s = dataset[i]
        lengths.append(s["sampled_num_frames"])
    lengths = np.array(lengths)
    print(f"Checked {len(lengths)} samples -> unique sampled lengths: {np.unique(lengths)}")
    assert np.all(lengths == SEQUENCE_LENGTH), "Mismatch: not all sequences have target length!"
    print("PASS: all sampled sequences have the fixed target length.")

print("\n=== Sequence length consistency check (train, 50 random samples) ===")
check_sequence_lengths(train_dataset, n=50)

# ---------------------------------------------------------
# DATALOADER (still batch_size=1 — resolutions differ until Step 5 resize)
# ---------------------------------------------------------
def raw_collate_fn(batch):
    return batch

train_loader = DataLoader(
    train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=raw_collate_fn
)

print("\n=== DataLoader smoke test (3 batches, batch_size=1) ===")
for i, batch in enumerate(train_loader):
    item = batch[0]
    print(f"Batch {i}: video={os.path.basename(item['video_path'])}, "
          f"class={item['class_name']}, "
          f"orig_frames={item['original_num_frames']}, "
          f"sampled_frames={item['frames'].shape}")
    if i == 2:
        break

print("\nStep 4 complete: fixed-length (90-frame) sequence sampling verified.")


# =========================================================
# NOTEBOOK CELL 4
# =========================================================

import cv2
import numpy as np

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
TARGET_SIZE = (224, 224)  # (width, height) for cv2.resize
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------------------------------------------------
# PREPROCESSING FUNCTION: resize + scale + normalize
# ---------------------------------------------------------
def preprocess_frames(frames, target_size=TARGET_SIZE, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    frames: (T, H, W, C) uint8 array, RGB
    Returns: (T, target_H, target_W, C) float32 array, normalized
    """
    T = frames.shape[0]
    target_w, target_h = target_size

    resized = np.empty((T, target_h, target_w, 3), dtype=np.float32)
    for i in range(T):
        # cv2.resize expects (width, height) and works on uint8 or float
        r = cv2.resize(frames[i], (target_w, target_h), interpolation=cv2.INTER_AREA)
        resized[i] = r

    # Scale to [0, 1]
    resized /= 255.0

    # Normalize per channel (ImageNet mean/std)
    resized = (resized - mean) / std

    return resized  # (T, target_h, target_w, 3), float32


# ---------------------------------------------------------
# EXTEND DATASET CLASS: adds preprocessing after sequence sampling
# ---------------------------------------------------------
class SignLanguageVideoDataset(Dataset):
    def __init__(self, dataframe, split, label_to_idx,
                 sequence_length=SEQUENCE_LENGTH, target_size=TARGET_SIZE):
        self.df = dataframe[dataframe["split"] == split].reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.sequence_length = sequence_length
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames read from video: {video_path}")

        return np.stack(frames, axis=0)  # (T, H, W, C) uint8

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        class_label = row["class_label"]

        raw_frames = self._load_video_frames(video_path)
        sampled_frames = uniform_temporal_sample(raw_frames, self.sequence_length)   # (90, H, W, C) uint8
        processed_frames = preprocess_frames(sampled_frames, self.target_size)       # (90, 224, 224, 3) float32

        label_idx = self.label_to_idx[class_label]

        sample = {
            "frames": processed_frames,
            "label": label_idx,
            "class_name": class_label,
            "video_path": video_path,
            "original_num_frames": raw_frames.shape[0],
            "sampled_num_frames": sampled_frames.shape[0]
        }
        return sample


# ---------------------------------------------------------
# REBUILD DATASETS WITH PREPROCESSING
# ---------------------------------------------------------
train_dataset = SignLanguageVideoDataset(df, split="train", label_to_idx=label_to_idx)
val_dataset   = SignLanguageVideoDataset(df, split="val",   label_to_idx=label_to_idx)
test_dataset  = SignLanguageVideoDataset(df, split="test",  label_to_idx=label_to_idx)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size:   {len(val_dataset)}")
print(f"Test dataset size:  {len(test_dataset)}")

# ---------------------------------------------------------
# TEST: SINGLE SAMPLE — VERIFY SHAPE, DTYPE, VALUE RANGE
# ---------------------------------------------------------
sample = train_dataset[0]
frames = sample["frames"]

print("\n=== Single sample check (train_dataset[0]) ===")
print("video_path:", sample["video_path"])
print("class_name:", sample["class_name"])
print("frames shape (T, H, W, C):", frames.shape)
print("frames dtype:", frames.dtype)
print("frames min/max:", frames.min(), frames.max())
print("frames mean (per channel):", frames.reshape(-1, 3).mean(axis=0))
print("frames std (per channel):", frames.reshape(-1, 3).std(axis=0))

# ---------------------------------------------------------
# TEST: CONSISTENCY ACROSS MULTIPLE SAMPLES
# ---------------------------------------------------------
def check_preprocessing_consistency(dataset, n=20):
    shapes = set()
    for i in np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False):
        s = dataset[i]
        shapes.add(s["frames"].shape)
    print(f"Checked {n} samples -> unique shapes found: {shapes}")
    assert len(shapes) == 1, "Mismatch: inconsistent shapes across samples!"
    print("PASS: all samples have identical shape after preprocessing.")

print("\n=== Preprocessing consistency check (train, 20 random samples) ===")
check_preprocessing_consistency(train_dataset, n=20)

print("\nStep 5 complete: resize + normalization verified.")
print("Note: batching still deferred to Step 6 (tensor conversion), where full DataLoader batching becomes possible.")


# =========================================================
# NOTEBOOK CELL 5
# =========================================================

import torch

# ---------------------------------------------------------
# EXTEND DATASET CLASS: adds tensor conversion (final step)
# ---------------------------------------------------------
class SignLanguageVideoDataset(Dataset):
    def __init__(self, dataframe, split, label_to_idx,
                 sequence_length=SEQUENCE_LENGTH, target_size=TARGET_SIZE):
        self.df = dataframe[dataframe["split"] == split].reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.sequence_length = sequence_length
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No frames read from video: {video_path}")

        return np.stack(frames, axis=0)  # (T, H, W, C) uint8

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        class_label = row["class_label"]

        raw_frames = self._load_video_frames(video_path)
        sampled_frames = uniform_temporal_sample(raw_frames, self.sequence_length)   # (90, H, W, C) uint8
        processed_frames = preprocess_frames(sampled_frames, self.target_size)       # (90, 224, 224, 3) float32

        # ---- Tensor conversion ----
        # numpy (T, H, W, C) -> torch (T, H, W, C) -> permute -> (T, C, H, W)
        frames_tensor = torch.from_numpy(processed_frames).permute(0, 3, 1, 2).contiguous()
        # frames_tensor shape: (T, C, H, W), dtype float32

        label_idx = self.label_to_idx[class_label]
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        sample = {
            "frames": frames_tensor,       # torch.FloatTensor (90, 3, 224, 224)
            "label": label_tensor,         # torch.LongTensor scalar
            "class_name": class_label,
            "video_path": video_path
        }
        return sample


# ---------------------------------------------------------
# REBUILD DATASETS WITH FULL PIPELINE
# ---------------------------------------------------------
train_dataset = SignLanguageVideoDataset(df, split="train", label_to_idx=label_to_idx)
val_dataset   = SignLanguageVideoDataset(df, split="val",   label_to_idx=label_to_idx)
test_dataset  = SignLanguageVideoDataset(df, split="test",  label_to_idx=label_to_idx)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size:   {len(val_dataset)}")
print(f"Test dataset size:  {len(test_dataset)}")

# ---------------------------------------------------------
# TEST: SINGLE SAMPLE
# ---------------------------------------------------------
sample = train_dataset[0]
print("\n=== Single sample check (train_dataset[0]) ===")
print("frames tensor shape (T, C, H, W):", sample["frames"].shape)
print("frames tensor dtype:", sample["frames"].dtype)
print("label tensor:", sample["label"], "| dtype:", sample["label"].dtype)
print("class_name:", sample["class_name"])

# ---------------------------------------------------------
# DEFAULT COLLATE NOW WORKS — every sample is identically shaped
# No custom collate_fn needed anymore; PyTorch will auto-stack.
# ---------------------------------------------------------
BATCH_SIZE = 8  # adjust later based on GPU memory once CNN is added

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# ---------------------------------------------------------
# TEST: FULL BATCH FROM DATALOADER
# ---------------------------------------------------------
print(f"\n=== DataLoader batch test (batch_size={BATCH_SIZE}) ===")
batch = next(iter(train_loader))

print("Batch frames shape (B, T, C, H, W):", batch["frames"].shape)
print("Batch frames dtype:", batch["frames"].dtype)
print("Batch labels shape:", batch["label"].shape)
print("Batch labels dtype:", batch["label"].dtype)
print("Batch labels values:", batch["label"].tolist())
print("Batch class names:", batch["class_name"])

# ---------------------------------------------------------
# FINAL SANITY CHECKS
# ---------------------------------------------------------
expected_shape = (BATCH_SIZE, SEQUENCE_LENGTH, 3, 224, 224)
assert tuple(batch["frames"].shape) == expected_shape, \
    f"Shape mismatch! Expected {expected_shape}, got {tuple(batch['frames'].shape)}"
assert batch["frames"].dtype == torch.float32, "Frames should be float32"
assert batch["label"].dtype == torch.int64, "Labels should be int64 (long)"

print("\nAll assertions passed.")
print("Step 6 complete: full preprocessing pipeline ready for CNN feature extraction.")
print(f"\nFinal tensor shape per batch: (Batch={BATCH_SIZE}, Time={SEQUENCE_LENGTH}, Channels=3, H=224, W=224)")


# =========================================================
# NOTEBOOK CELL 6
# =========================================================

import torch
import torch.nn as nn
import torchvision.models as models

# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------------------------------------
# CNN FEATURE EXTRACTOR (ResNet50, frozen, pretrained on ImageNet)
# ---------------------------------------------------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = resnet.fc.in_features

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.freeze = freeze

    def forward(self, x):
        """x: (N, C, H, W) -> (N, feature_dim)"""
        features = self.backbone(x)
        return features.flatten(start_dim=1)


class SequenceCNNFeatureExtractor(nn.Module):
    def __init__(self, cnn_extractor, chunk_size=32):
        super().__init__()
        self.cnn = cnn_extractor
        self.chunk_size = chunk_size

    def forward(self, x):
        """x: (B, T, C, H, W) -> (B, T, feature_dim)"""
        B, T, C, H, W = x.shape
        assert C == 3, f"Expected channel dim=3 at position 2, got shape {tuple(x.shape)}"

        x_flat = x.view(B * T, C, H, W)
        all_features = []
        with torch.no_grad():
            for start in range(0, x_flat.shape[0], self.chunk_size):
                end = start + self.chunk_size
                chunk = x_flat[start:end]
                feat_chunk = self.cnn(chunk)
                all_features.append(feat_chunk.cpu())
                del chunk, feat_chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        features_flat = torch.cat(all_features, dim=0)
        return features_flat.view(B, T, self.cnn.feature_dim)


# ---------------------------------------------------------
# INSTANTIATE
# ---------------------------------------------------------
cnn_backbone = CNNFeatureExtractor(freeze=True).to(device)
sequence_extractor = SequenceCNNFeatureExtractor(cnn_backbone, chunk_size=32)

print(f"\nBackbone: ResNet50 (ImageNet pretrained)")
print(f"Feature dimension per frame: {cnn_backbone.feature_dim}")
trainable_params = sum(p.numel() for p in cnn_backbone.backbone.parameters() if p.requires_grad)
print(f"Trainable backbone params: {trainable_params:,} (should be 0)")

# ---------------------------------------------------------
# FETCH ONE BATCH
# ---------------------------------------------------------
raw_batch = next(iter(train_loader))

if isinstance(raw_batch, list):
    VERIFY_SUBSET = min(2, len(raw_batch))
    subset = raw_batch[:VERIFY_SUBSET]
    input_tensor = torch.stack([item["frames"] for item in subset], dim=0)
    labels = torch.stack([item["label"] for item in subset], dim=0)
    class_names = [item["class_name"] for item in subset]
elif isinstance(raw_batch, dict):
    VERIFY_SUBSET = min(2, raw_batch["frames"].shape[0])
    input_tensor = raw_batch["frames"][:VERIFY_SUBSET]
    labels = raw_batch["label"][:VERIFY_SUBSET]
    class_names = raw_batch["class_name"][:VERIFY_SUBSET]
else:
    raise TypeError(f"Unexpected batch type from train_loader: {type(raw_batch)}")

print("\n=== Raw input tensor, straight from DataLoader (before any conversion) ===")
print("Shape:", tuple(input_tensor.shape))
print(f"  dim[2] = {input_tensor.shape[2]}   (if this is 3, it's already channels-first: B,T,C,H,W)")
print(f"  dim[4] = {input_tensor.shape[4]}   (if this is 3, it's channels-last: B,T,H,W,C)")

# ---------------------------------------------------------
# AUTO-DETECT CHANNEL POSITION AND CONVERT ONLY IF NEEDED
# ---------------------------------------------------------
if input_tensor.shape[2] == 3 and input_tensor.shape[4] != 3:
    print("\nDetected: already (B, T, C, H, W). No permute needed.")
elif input_tensor.shape[4] == 3 and input_tensor.shape[2] != 3:
    print("\nDetected: (B, T, H, W, C). Applying permute(0, 1, 4, 2, 3) -> (B, T, C, H, W).")
    input_tensor = input_tensor.permute(0, 1, 4, 2, 3).contiguous()
else:
    raise ValueError(
        f"Cannot unambiguously detect channel position from shape {tuple(input_tensor.shape)}. "
        f"Both dim[2] and dim[4] are ambiguous (3 or not-3 in both) -- inspect manually."
    )

input_tensor = input_tensor.to(device)

print("\n=== Input tensor fed to CNN (B, T, C, H, W) ===")
print("Shape:", tuple(input_tensor.shape))
assert input_tensor.shape[2] == 3, f"Still wrong after conversion: {tuple(input_tensor.shape)}"
print("Shape check PASSED: channels at position 2.")

# ---------------------------------------------------------
# RUN CNN FEATURE EXTRACTION
# ---------------------------------------------------------
feature_sequence = sequence_extractor(input_tensor)

print("\n=== Output feature tensor (B, T, feature_dim) ===")
print("Shape:", tuple(feature_sequence.shape))
print("Feature value range (min/max):", feature_sequence.min().item(), feature_sequence.max().item())

# ---------------------------------------------------------
# TEMPORAL ORDER CHECK
# ---------------------------------------------------------
video_idx = 0
single_video_frames = input_tensor[video_idx]
with torch.no_grad():
    individually_extracted = cnn_backbone(single_video_frames).cpu()
batched_result = feature_sequence[video_idx]
max_diff = (individually_extracted - batched_result).abs().max().item()
print(f"\nMax diff (individual vs chunked): {max_diff:.8f}")
print("PASS" if max_diff < 1e-4 else "WARNING: mismatch")

del input_tensor, single_video_frames
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\nStep complete.")


# =========================================================
# NOTEBOOK CELL 7
# =========================================================

import torch
import torch.nn as nn

# ---------------------------------------------------------
# CONFIG (pulled from your actual pipeline, not assumed)
# ---------------------------------------------------------
NUM_CLASSES = len(label_to_idx)
CNN_FEATURE_DIM = cnn_backbone.feature_dim   # 2048, from your Cell 6 ResNet50 extractor

print(f"NUM_CLASSES: {NUM_CLASSES}")
print(f"CNN_FEATURE_DIM (LSTM input_size): {CNN_FEATURE_DIM}")

# ---------------------------------------------------------
# LSTM TEMPORAL MODEL
# Consumes the (B, T, feature_dim) sequence from your
# SequenceCNNFeatureExtractor and classifies into NUM_CLASSES.
# Many-to-one: uses the final hidden state as the video-level
# representation, since each video has exactly one label.
# ---------------------------------------------------------
class LSTMTemporalModel(nn.Module):
    def __init__(self, input_size=CNN_FEATURE_DIM, hidden_size=256,
                 num_layers=1, num_classes=NUM_CLASSES, bidirectional=False):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # matches your (B, T, feature_dim) layout
            bidirectional=bidirectional
        )
        self.classifier = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        """
        x: (B, T, feature_dim)
        Returns: logits (B, num_classes), lstm_out (B, T, hidden_size)
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers * num_directions, B, hidden_size)
        if self.num_directions == 2:
            final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            final_hidden = h_n[-1]                       # (B, hidden_size)

        logits = self.classifier(final_hidden)           # (B, num_classes)
        return logits, lstm_out


# ---------------------------------------------------------
# INSTANTIATE
# feature_sequence (from Cell 6) already lives on CPU -- your
# SequenceCNNFeatureExtractor moves chunk outputs there during
# extraction. Building/running this model on CPU keeps everything
# on the same device with no extra transfer, and also avoids
# retriggering the P100/CUDA kernel mismatch from earlier for
# this forward-pass-only check.
# ---------------------------------------------------------
lstm_model = LSTMTemporalModel().to("cpu")
lstm_model.eval()

lstm_trainable_params = sum(p.numel() for p in lstm_model.parameters() if p.requires_grad)
print(f"\nLSTM model trainable params: {lstm_trainable_params:,} (untrained -- your friend trains these)")

# ---------------------------------------------------------
# FORWARD-PASS TEST using your ACTUAL feature_sequence from Cell 6
# ---------------------------------------------------------
print("\n=== Input to LSTM (your actual feature_sequence) ===")
print("Shape (B, T, feature_dim):", tuple(feature_sequence.shape))
print(f"  B = {feature_sequence.shape[0]}  -> batch size (subset from Cell 6)")
print(f"  T = {feature_sequence.shape[1]}  -> sequence length (90 frames)")
print(f"  feature_dim = {feature_sequence.shape[2]}  -> matches LSTM input_size (2048)")

with torch.no_grad():
    logits, lstm_out = lstm_model(feature_sequence)

print("\n=== LSTM outputs ===")
print("lstm_out shape (B, T, hidden_size):", tuple(lstm_out.shape))
print("logits shape (B, num_classes):", tuple(logits.shape))

assert logits.shape == (feature_sequence.shape[0], NUM_CLASSES), \
    f"Shape mismatch: expected ({feature_sequence.shape[0]}, {NUM_CLASSES}), got {tuple(logits.shape)}"
print("Shape check PASSED.")

predicted_idx = torch.argmax(logits, dim=1)
predicted_classes = [idx_to_label[i.item()] for i in predicted_idx]
print("\nPredicted classes (UNTRAINED random weights -- shape check only, not meaningful):", predicted_classes)
print("Actual ground-truth classes for this subset:", class_names)

print("\nLSTM forward-pass verification complete. Model is NOT trained.")


# =========================================================
# NOTEBOOK CELL 8
# =========================================================

# ---------------------------------------------------------
# GRU TEMPORAL MODEL
# Same role as LSTMTemporalModel, kept as a separate class so
# both can be compared later. GRU has no cell state (only h_n),
# unlike LSTM.
# ---------------------------------------------------------
class GRUTemporalModel(nn.Module):
    def __init__(self, input_size=CNN_FEATURE_DIM, hidden_size=256,
                 num_layers=1, num_classes=NUM_CLASSES, bidirectional=False):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.classifier = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        """
        x: (B, T, feature_dim)
        Returns: logits (B, num_classes), gru_out (B, T, hidden_size)
        """
        gru_out, h_n = self.gru(x)
        if self.num_directions == 2:
            final_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            final_hidden = h_n[-1]

        logits = self.classifier(final_hidden)
        return logits, gru_out


gru_model = GRUTemporalModel().to("cpu")
gru_model.eval()

gru_trainable_params = sum(p.numel() for p in gru_model.parameters() if p.requires_grad)
print(f"GRU model trainable params: {gru_trainable_params:,} (untrained)")

print("\n=== Input to GRU (same feature_sequence) ===")
print("Shape (B, T, feature_dim):", tuple(feature_sequence.shape))

with torch.no_grad():
    gru_logits, gru_out = gru_model(feature_sequence)

print("\n=== GRU outputs ===")
print("gru_out shape (B, T, hidden_size):", tuple(gru_out.shape))
print("logits shape (B, num_classes):", tuple(gru_logits.shape))

assert gru_logits.shape == (feature_sequence.shape[0], NUM_CLASSES), \
    f"Shape mismatch: expected ({feature_sequence.shape[0]}, {NUM_CLASSES}), got {tuple(gru_logits.shape)}"
print("Shape check PASSED.")

print("\nGRU forward-pass verification complete. Model is NOT trained.")
