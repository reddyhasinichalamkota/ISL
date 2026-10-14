import os
from typing import Any, Dict
from src.utils.logger import get_logger

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logger = get_logger("ConfigLoader")

DEFAULT_CONFIG = {
    "camera": {"index": 0, "width": 640, "height": 480, "fps": 30},
    "preprocessing": {"image_width": 224, "image_height": 224, "channels": 3, "normalization": True},
    "dynamic": {"sequence_length": 30, "inference_interval": 5},
    "model": {"backbone": "MobileNetV2", "temporal_model": "GRU"},
    "inference": {"confidence_threshold": 0.70, "smoothing": True, "smoothing_window_size": 5},
    "training": {"batch_size": 32, "epochs": 30, "learning_rate": 0.001, "seed": 42}
}

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads configuration from YAML file or returns defaults."""
    if not HAS_YAML or not os.path.exists(config_path):
        if not HAS_YAML:
            logger.warning("PyYAML not installed. Falling back to default configuration.")
        else:
            logger.warning(f"Config file not found at {config_path}. Falling back to default settings.")
        return DEFAULT_CONFIG
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return cfg if cfg else DEFAULT_CONFIG
    except Exception as e:
        logger.error(f"Error reading config file {config_path}: {e}")
        return DEFAULT_CONFIG
