import os
from PIL import Image

DATASET_DIR = r"C:\Users\reddy\OneDrive\Desktop\ISL"

count = 0

for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        path = os.path.join(root, file)

        if file.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
        ):
            try:
                with Image.open(path) as img:
                    img.verify()
            except Exception:
                print("CORRUPTED IMAGE:", path)
                count += 1

print("\nTotal corrupted images:", count)