import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Tuple
from src.models.cnn_model import build_cnn_feature_extractor

def build_gru_dynamic_model(
    sequence_length: int = 30,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 20,
    backbone: str = "MobileNetV2"
) -> tf.keras.Model:
    """
    Builds dynamic sign recognition model using CNN feature extraction + GRU temporal modeling.
    Input Shape: (Batch, Sequence_Length, Height, Width, Channels).
    """
    sequence_inputs = layers.Input(shape=(sequence_length,) + input_shape)

    cnn_extractor = build_cnn_feature_extractor(input_shape=input_shape, backbone=backbone, embedding_dim=512)
    feature_sequences = layers.TimeDistributed(cnn_extractor)(sequence_inputs)

    # Temporal GRU layers
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(feature_sequences)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=False))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dynamic_gru_output")(x)

    model = models.Model(inputs=sequence_inputs, outputs=outputs, name="Dynamic_ISL_GRU_Model")
    return model
