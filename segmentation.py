"""
ISL Static Sign Preprocessing Pipeline
---------------------------------------
Steps (per your notes):
1. Load image using OpenCV
2. Detect the hand using MediaPipe
3. Find min/max X/Y coordinates of the landmarks
4. Create a bounding box around the hand
5. Crop the hand from the image
6. Resize it to 224 x 224
7. Save the cropped image so it can be used for CNN training

Run this once on your Dataset/{Training,Validation,Testing} folders.
It creates a mirrored folder structure with "_cropped" appended,
containing only the hand-cropped images. Any image where MediaPipe
fails to detect a hand is skipped and logged (so you can review/remove
those originals or retake them).

Install requirements first:
    pip install opencv-python mediapipe --break-system-packages
"""

import os
import cv2
import mediapipe as mp

# ---------------- CONFIG ----------------
INPUT_ROOT = "Dataset"              # contains Training/, Validation/, Testing/
OUTPUT_ROOT = "Dataset_cropped"     # mirrored structure will be created here
IMG_SIZE = 224
PADDING = 20                        # extra pixels around the tight bbox so fingertips aren't clipped
SPLITS = ["Training", "Validation", "Testing"]
# -----------------------------------------

mp_hands = mp.solutions.hands


def get_hand_bbox(image_rgb, hands_detector, img_w, img_h):
    """Run MediaPipe on one image, return (xmin, ymin, xmax, ymax) in pixels, or None."""
    results = hands_detector.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None

    # If multiple hands are detected, just use the first one
    landmarks = results.multi_hand_landmarks[0].landmark

    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]

    xmin, xmax = int(min(xs)), int(max(xs))
    ymin, ymax = int(min(ys)), int(max(ys))

    # Add padding, clamped to image bounds
    xmin = max(0, xmin - PADDING)
    ymin = max(0, ymin - PADDING)
    xmax = min(img_w, xmax + PADDING)
    ymax = min(img_h, ymax + PADDING)

    return xmin, ymin, xmax, ymax


def process_image(img_path, hands_detector):
    """Load -> detect -> crop -> resize. Returns the processed image or None if no hand found."""
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        return None

    img_h, img_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    bbox = get_hand_bbox(image_rgb, hands_detector, img_w, img_h)
    if bbox is None:
        return None

    xmin, ymin, xmax, ymax = bbox
    cropped = image_bgr[ymin:ymax, xmin:xmax]

    if cropped.size == 0:
        return None

    resized = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
    return resized


def process_dataset():
    skipped_log = []
    total, saved = 0, 0

    # static_image_mode=True because we process one image at a time (not video frames)
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands_detector:

        for split in SPLITS:
            split_in = os.path.join(INPUT_ROOT, split)
            if not os.path.isdir(split_in):
                print(f"Skipping missing split folder: {split_in}")
                continue

            for class_name in sorted(os.listdir(split_in)):
                class_in = os.path.join(split_in, class_name)
                if not os.path.isdir(class_in):
                    continue

                class_out = os.path.join(OUTPUT_ROOT, split, class_name)
                os.makedirs(class_out, exist_ok=True)

                for fname in os.listdir(class_in):
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue

                    total += 1
                    img_path = os.path.join(class_in, fname)
                    result = process_image(img_path, hands_detector)

                    if result is None:
                        skipped_log.append(img_path)
                        continue

                    out_path = os.path.join(class_out, fname)
                    cv2.imwrite(out_path, result)
                    saved += 1

                print(f"[{split}/{class_name}] done.")

    print(f"\nTotal images: {total} | Saved: {saved} | Skipped (no hand found): {len(skipped_log)}")

    if skipped_log:
        with open("skipped_images.txt", "w") as f:
            f.write("\n".join(skipped_log))
        print("List of skipped images written to skipped_images.txt")


if __name__ == "__main__":
    process_dataset()