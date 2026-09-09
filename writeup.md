# Step to do for Review 1
## Find a suitable static ISL dataset — you already have the RealSign dataset (A–Z structure), so this can mostly be "confirm it's suitable."

## Analyze the dataset: number of classes, images per class, resolution/quality of images.

## Check data quality: look for corrupted or duplicate images, and check class imbalance (are some letters underrepresented?).

## Select initial classes — confirm you're doing all A–Z or a subset for the first pass.

## Research CNN and transfer learning — enough to justify your model choice (MobileNetV2) in the review.

## Build a small preprocessing prototype — a scrip

## Step - 1:
   ### Dataset Analysis and Preprocessing 

**What I did:**
I analyzed the ISL dataset using Python to check the number of images in each alphabet class, corrupted images, duplicate images, and class distribution. I also tested preprocessing by resizing an image to 224 × 224 pixels and normalizing its pixel values.

**Output:**
`Training:` 18,198 images
`Testing:` 5,198 images
`Validation:` 2,579 images
`Classes:` 26 (A–Z)
`Corrupted images:` 0
| Duplicate groups: 141 in Training |
| Duplicate groups: 36 in Testing |
| Duplicate groups: 10 in Validation |
Preprocessed image shape: (224, 224, 3)
Normalized pixel range: approximately 0.024–0.827
Class distribution and sample-image plots were successfully generated.

**Result:**
The dataset was successfully analyzed and the preprocessing prototype worked correctly.

~`Duplicate check found 141 duplicate image groups in Training, 36 in Testing, and 10 in Validation. These were left as-is since duplicates within the same split don't cause data leakage — they just mean slightly less variety in that split. No corrupted images remain after removing 2 corrupted files from Testing.`~

# Cross-Split Leakage Check
Ran isl_leakage_check.py to check if the same image appeared in more than one split (Training/Testing/Validation). Found 648 leaked images shared between Training and Testing/Validation out of 25,085 total images.

# after python isl_leakage_check.py
`After isl_leakage_check.py found 648 leaked images, you ran isl_remove_train_leaks.py to delete them from Training. That removed 685 images total. Training went from 18,198 → 17,513 images.` 

# after isl_remove_train_leaks.py
Ran isl_remove_train_leaks.py to remove these duplicates from Training only (kept Testing/Validation as-is). This removed 685 images from Training.

# Completed data cleaning
`A second leakage check found 51 remaining leaks, this time between Testing and Validation (missed by the first cleanup, which only checked Training). 52 leaked images were removed from Validation, keeping Testing as the untouched final holdout set. A final leakage check confirmed 0 cross-split leakage across all three splits. Final dataset: Training 17,513 images, Testing 5,198 images, Validation 2,527 images, all 26 classes represented in each split with minor class imbalance (documented, to be addressed in Review 2 via class weighting/augmentation).`

> Initial dataset
→ Cross-split leakage detected (648 leaked images)
→ Training leakage removed (685 images)
→ Testing–Validation leakage checked separately
→ 52 images removed from Validation
→ Final leakage check
→ 0 cross-split leakage ✅

>🧹 Final cleaning status
| Split          |     Images |  Classes |
| -------------- | ---------: | -------: |
| **Training**   | **17,513** | 26 (A–Z) |
| **Validation** |  **2,527** | 26 (A–Z) |
| **Testing**    |  **5,198** | 26 (A–Z) |
| **Total**      | **25,238** |   **26** |
→ Corrupted images: 0 remaining
→ Cross-split leakage: 0
→ Training–Testing/Validation leakage: removed
→ Testing–Validation leakage: removed
→ All 26 classes: represented in all three splits
→ Minor class imbalance: present
