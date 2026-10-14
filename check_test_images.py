from PIL import Image
import os

root = "Dataset/Testing"

bad_files = []
total = 0

for folder, _, files in os.walk(root):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")):

            path = os.path.join(folder, file)
            total += 1

            try:
                with Image.open(path) as img:
                    img.verify()

            except Exception as e:
                bad_files.append((path, str(e)))

print("\n==============================")
print("TEST DATASET IMAGE CHECK")
print("==============================")

print("Total image files:", total)
print("Bad image files:", len(bad_files))

if bad_files:
    print("\nBAD FILES:")
    for path, error in bad_files:
        print("\n", path)
        print("Error:", error)
else:
    print("\nAll images are valid!")