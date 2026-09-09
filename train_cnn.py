import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 1. Dataset paths
# ============================================================

train_dir = "Dataset/Training"
val_dir = "Dataset/Validation"
test_dir = "Dataset/Testing"


# ============================================================
# 2. Image settings
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 26
EPOCHS = 10


# ============================================================
# 3. Load datasets
# ============================================================

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
# 4. Get class names
# ============================================================

class_names = train_data.class_names

print("\n==============================")
print("CLASS NAMES")
print("==============================")

print(class_names)

print("\nNumber of classes:", len(class_names))


# ============================================================
# 5. Normalize images
# ============================================================

normalization = layers.Rescaling(1.0 / 255)


# ============================================================
# 6. Build CNN
# ============================================================

model = models.Sequential([

    normalization,

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# 7. Compile model
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 8. Display model
# ============================================================

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
# 10. Training and validation results
# ============================================================

train_accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]


print("\n==============================")
print("TRAINING RESULTS")
print("==============================")

print(
    "Final Training Accuracy:",
    train_accuracy[-1]
)

print(
    "Final Validation Accuracy:",
    val_accuracy[-1]
)

print(
    "Final Training Loss:",
    train_loss[-1]
)

print(
    "Final Validation Loss:",
    val_loss[-1]
)


# ============================================================
# 11. Test model
# ============================================================

test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=1
)


print("\n==============================")
print("TEST RESULTS")
print("==============================")

print(
    "Test Accuracy:",
    test_accuracy
)

print(
    "Test Loss:",
    test_loss
)


# ============================================================
# 12. Get predictions
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

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_classes
    )


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ============================================================
# 13. Precision
# ============================================================

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# 14. Recall
# ============================================================

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# 15. F1 Score
# ============================================================

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# 16. Print evaluation metrics
# ============================================================

print("\n==============================")
print("EVALUATION METRICS")
print("==============================")

print(
    f"Accuracy  : {test_accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1-Score  : {f1:.4f}"
)


# ============================================================
# 17. Classification Report
# ============================================================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print(report)


# ============================================================
# 18. Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

print(cm)


# ============================================================
# 19. Plot Confusion Matrix
# ============================================================

plt.figure(figsize=(12, 10))

plt.imshow(cm)

plt.title("Confusion Matrix")

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
    "confusion_matrix.png",
    dpi=300
)

plt.show()


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
    "accuracy_curve.png",
    dpi=300
)

plt.show()


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
    "loss_curve.png",
    dpi=300
)

plt.show()


# ============================================================
# 22. Save model
# ============================================================

model.save(
    "model/custom_cnn.keras"
)


# ============================================================
# 23. Final summary
# ============================================================

print("\n==============================")
print("FINAL MODEL SUMMARY")
print("==============================")

print(
    f"Training Accuracy   : {train_accuracy[-1]:.4f}"
)

print(
    f"Validation Accuracy : {val_accuracy[-1]:.4f}"
)

print(
    f"Test Accuracy       : {test_accuracy:.4f}"
)

print(
    f"Test Loss           : {test_loss:.4f}"
)

print(
    f"Precision           : {precision:.4f}"
)

print(
    f"Recall              : {recall:.4f}"
)

print(
    f"F1-Score            : {f1:.4f}"
)

print("\nModel saved successfully!")

print("Confusion matrix saved as: confusion_matrix.png")

print("Accuracy curve saved as: accuracy_curve.png")

print("Loss curve saved as: loss_curve.png")