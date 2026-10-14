import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import tensorflow as tf
from src.utils.config import load_config
from src.utils.class_mapping import ClassMappingManager
from src.utils.logger import get_logger
from src.models.dynamic_model import build_dynamic_model

logger = get_logger("TrainDynamic")

def run_dynamic_training(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    train_cfg = cfg.get("training", {})
    prep_cfg = cfg.get("preprocessing", {})
    model_cfg = cfg.get("model", {})
    dyn_cfg = cfg.get("dynamic", {})

    out_dir = "models/dynamic"
    os.makedirs(out_dir, exist_ok=True)

    seq_len = dyn_cfg.get("sequence_length", 30)
    t_model = model_cfg.get("temporal_model", "GRU")
    img_size = (prep_cfg.get("image_height", 224), prep_cfg.get("image_width", 224))

    logger.info(f"Setting up dynamic training pipeline for architecture: CNN + {t_model} (SeqLen={seq_len})")

    # Dynamic model builder validation
    model = build_dynamic_model(
        temporal_model_type=t_model,
        sequence_length=seq_len,
        input_shape=img_size + (3,),
        num_classes=20,
        backbone=model_cfg.get("backbone", "MobileNetV2")
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=train_cfg.get("learning_rate", 0.001)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model_path = os.path.join(out_dir, "dynamic_model.keras")
    model.save(model_path)
    logger.info(f"Dynamic model architecture initialized and compiled to {model_path}")

if __name__ == "__main__":
    run_dynamic_training()
