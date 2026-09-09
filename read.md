# Review 1 — ISL Dataset Analysis and Preprocessing

## Steps to Complete for Review 1

### step 1. Select a Suitable Static ISL Dataset

* Use the existing **RealSign dataset**, which contains an **A–Z alphabet structure**.
* Confirm that the dataset is suitable for static Indian Sign Language (ISL) recognition.

### step 2. Analyze the Dataset

Analyze the following:

* Number of classes
* Number of images per class
* Image resolution and quality
* Class distribution

### step 3. Check Data Quality

Check for:

* Corrupted images
* Duplicate images
* Cross-split data leakage
* Class imbalance

### step 4. Select Initial Classes

* Confirm whether the first implementation will use **all 26 classes (A–Z)** or a subset.

### step 5. Research CNN and Transfer Learning

* Study CNN-based image classification.
* Research transfer learning.
* Justify the selection of **MobileNetV2** as the initial model for the project.

### step 6. Build a Preprocessing Prototype

Develop a small preprocessing script to:

* Resize images to **224 × 224 pixels**
* Normalize pixel values
* Verify image shape and preprocessing output

---

# Step 1: Dataset Analysis and Preprocessing

## What I Did

I analyzed the ISL dataset using Python to determine:

* The number of images in each alphabet class
* The number of classes
* The presence of corrupted images
* Duplicate images within each dataset split
* Class distribution
* Image quality and resolution

I also tested the preprocessing pipeline by:

* Resizing an image to **224 × 224 pixels**
* Normalizing its pixel values

---

## Initial Dataset Analysis

| Split      | Number of Images |
| ---------- | ---------------: |
| Training   |           18,198 |
| Testing    |            5,198 |
| Validation |            2,579 |
| **Total**  |       **25,975** |

* **Number of Classes:** 26 (A–Z)
* **Corrupted Images:** 2 corrupted files were identified in Testing and removed.
* **Preprocessed Image Shape:** `(224, 224, 3)`
* **Normalized Pixel Range:** Approximately `0.024 – 0.827`

### Duplicate Groups

| Dataset Split | Duplicate Groups |
| ------------- | ---------------: |
| Training      |              141 |
| Testing       |               36 |
| Validation    |               10 |

Class-distribution plots and sample-image visualizations were also successfully generated.

---

## Initial Result

The dataset was successfully analyzed, and the preprocessing prototype worked correctly.

Duplicate images found within the same split were not considered data leakage. However, **cross-split duplicates** were checked separately because they can cause data leakage and lead to misleading model performance.

---

# Cross-Split Leakage Check

The script `isl_leakage_check.py` was used to determine whether identical images appeared across the **Training, Testing, and Validation** splits.

### Initial Result

* **648 leaked images** were identified between Training and Testing/Validation.
* These images were found among a total of **25,085 images** checked during the leakage analysis.

Because cross-split duplicates can cause **data leakage**, they were removed from the Training set.

---

# Training Leakage Removal

After identifying the leaked images, the script:

`isl_remove_train_leaks.py`

was executed.

### Result

* Training images before removal: **18,198**
* Training images after removal: **17,513**
* Total images removed from Training: **685**

The Testing and Validation sets were kept unchanged at this stage.

---

# Final Data Cleaning

A second leakage check was performed after cleaning the Training set.

### Result

The second check identified **51 remaining cross-split leaks**, specifically between:

* Testing
* Validation

To preserve the Testing set as the **final untouched holdout set**, the leaked images were removed from Validation only.

### Final Cleaning Result

* **52 images** were removed from Validation.
* Testing was kept unchanged.
* A final leakage check was performed after the cleanup.

### Final Leakage Check

The final check confirmed:

> **0 cross-split leakage across Training, Testing, and Validation.**

---

# Final Dataset

| Split      | Final Number of Images | Status                  |
| ---------- | ---------------------: | ----------------------- |
| Training   |             **17,513** | Cleaned                 |
| Testing    |              **5,198** | Untouched final holdout |
| Validation |              **2,527** | Cleaned                 |
| **Total**  |             **25,238** | **Final dataset**       |

Additional observations:

* **26 classes (A–Z)** are represented in each split.
* A **minor class imbalance** is present.
* The class imbalance will be addressed in **Review 2** using techniques such as:

  * Class weighting
  * Data augmentation

---

# Conclusion

The ISL dataset was successfully analyzed and cleaned.

The final dataset contains **25,238 images across 26 alphabet classes (A–Z)**. Corrupted images were removed, duplicate images were analyzed, and cross-split data leakage was eliminated.

The preprocessing pipeline was also successfully tested using **224 × 224 × 3** images with normalized pixel values.

The cleaned dataset is now ready for the next stage of the project: **CNN-based model development and transfer learning using MobileNetV2**.
