import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# ============================================================
# 1. Create output folders
# ============================================================

os.makedirs("model", exist_ok=True)
os.makedirs("evaluation_results", exist_ok=True)

# ============================================================
# 2. Dataset paths
# ============================================================

train_dir = "Dataset/Training"
val_dir = "Dataset/Validation"
test_dir = "Dataset/Testing"

# ============================================================
# 3. Image settings
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 26
EPOCHS = 30

# ============================================================
# 4. Load datasets
# ============================================================

print("\n==============================")
print("LOADING DATASETS")
print("==============================")

train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

val_data = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# ============================================================
# 5. Class names
# ============================================================

class_names = train_data.class_names

print("\n==============================")
print("CLASS NAMES")
print("==============================")

print(class_names)
print("Number of classes:", len(class_names))

# ============================================================
# 6. Normalize images
# ============================================================

normalization = layers.Rescaling(1.0 / 255)

# ============================================================
# 7. Build CNN
# ============================================================

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),

    normalization,

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.5),

    layers.Dense(NUM_CLASSES, activation="softmax")
])

# ============================================================
# 8. Compile model
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n==============================")
print("MODEL SUMMARY")
print("==============================")

model.summary()

# ============================================================
# 9. Train model
# ============================================================

print("\n==============================")
print("TRAINING MODEL")
print("==============================")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ============================================================
# 10. Training / Validation results
# ============================================================

train_accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

print("\n==============================")
print("TRAINING RESULTS")
print("==============================")

print(f"Final Training Accuracy   : {train_accuracy[-1]:.4f}")
print(f"Final Validation Accuracy : {val_accuracy[-1]:.4f}")
print(f"Final Training Loss       : {train_loss[-1]:.4f}")
print(f"Final Validation Loss     : {val_loss[-1]:.4f}")

# ============================================================
# 11. Test evaluation
# ============================================================

print("\n==============================")
print("TEST EVALUATION")
print("==============================")

test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=1
)

print(f"\nTest Accuracy : {test_accuracy:.4f}")
print(f"Test Loss     : {test_loss:.4f}")

# ============================================================
# 12. Generate predictions
# ============================================================

print("\n==============================")
print("GENERATING PREDICTIONS")
print("==============================")

y_true = []
y_pred = []

for images, labels in test_data:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("Total test samples evaluated:", len(y_true))

# ============================================================
# 13. Accuracy from predictions
# ============================================================

prediction_accuracy = accuracy_score(
    y_true,
    y_pred
)

# ============================================================
# 14. Weighted metrics
# ============================================================

weighted_precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

# ============================================================
# 15. Macro metrics
# ============================================================

macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

# ============================================================
# 16. Print evaluation metrics
# ============================================================

print("\n==============================")
print("EVALUATION METRICS")
print("==============================")

print(f"Test Accuracy       : {prediction_accuracy:.4f}")
print(f"Test Loss           : {test_loss:.4f}")

print(f"\nWeighted Precision  : {weighted_precision:.4f}")
print(f"Weighted Recall     : {weighted_recall:.4f}")
print(f"Weighted F1-Score   : {weighted_f1:.4f}")

print(f"\nMacro Precision     : {macro_precision:.4f}")
print(f"Macro Recall        : {macro_recall:.4f}")
print(f"Macro F1-Score      : {macro_f1:.4f}")

# ============================================================
# 17. Classification Report
# ============================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_text = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print(report_text)

# Save classification report
import pandas as pd

report_df = pd.DataFrame(report_dict).transpose()

report_df.to_csv(
    "evaluation_results/classification_report.csv"
)

# ============================================================
# 18. Confusion Matrix
# ============================================================

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)

# Save confusion matrix CSV
cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)

cm_df.to_csv(
    "evaluation_results/confusion_matrix.csv"
)

# ============================================================
# 19. Plot Confusion Matrix
# ============================================================

plt.figure(figsize=(12, 10))

plt.imshow(cm)

plt.title("CNN Confusion Matrix")

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.colorbar()

plt.xticks(
    np.arange(len(class_names)),
    class_names,
    rotation=90
)

plt.yticks(
    np.arange(len(class_names)),
    class_names
)

plt.tight_layout()

plt.savefig(
    "evaluation_results/confusion_matrix.png",
    dpi=300
)

plt.close()

# ============================================================
# 20. Accuracy Curve
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    train_accuracy,
    label="Training Accuracy"
)

plt.plot(
    val_accuracy,
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "evaluation_results/accuracy_curve.png",
    dpi=300
)

plt.close()

# ============================================================
# 21. Loss Curve
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    train_loss,
    label="Training Loss"
)

plt.plot(
    val_loss,
    label="Validation Loss"
)

plt.title("Training and Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    "evaluation_results/loss_curve.png",
    dpi=300
)

plt.close()

# ============================================================
# 22. Save all metrics to CSV
# ============================================================

metrics = {
    "Final Training Accuracy": train_accuracy[-1],
    "Final Validation Accuracy": val_accuracy[-1],
    "Final Training Loss": train_loss[-1],
    "Final Validation Loss": val_loss[-1],
    "Test Accuracy": prediction_accuracy,
    "Test Loss": test_loss,
    "Weighted Precision": weighted_precision,
    "Weighted Recall": weighted_recall,
    "Weighted F1-Score": weighted_f1,
    "Macro Precision": macro_precision,
    "Macro Recall": macro_recall,
    "Macro F1-Score": macro_f1
}

metrics_df = pd.DataFrame(
    list(metrics.items()),
    columns=["Metric", "Value"]
)

metrics_df.to_csv(
    "evaluation_results/evaluation_metrics.csv",
    index=False
)

# ============================================================
# 23. Save model
# ============================================================

model.save(
    "model/custom_cnn.keras"
)

# ============================================================
# 24. Final summary
# ============================================================

print("\n==============================")
print("FINAL MODEL SUMMARY")
print("==============================")

for metric, value in metrics.items():
    print(f"{metric:<30}: {value:.4f}")

print("\n==============================")
print("FILES GENERATED")
print("==============================")

print("Model:")
print("  model/custom_cnn.keras")

print("\nEvaluation:")
print("  evaluation_results/evaluation_metrics.csv")
print("  evaluation_results/classification_report.csv")
print("  evaluation_results/confusion_matrix.csv")
print("  evaluation_results/confusion_matrix.png")
print("  evaluation_results/accuracy_curve.png")
print("  evaluation_results/loss_curve.png")

print("\nMODEL TRAINING AND EVALUATION COMPLETED SUCCESSFULLY!")