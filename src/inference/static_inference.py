import os
import cv2
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, List
from src.preprocessing.image_preprocessing import preprocess_static_image
from src.utils.class_mapping import ClassMappingManager
from src.utils.logger import get_logger

logger = get_logger("StaticInferenceEngine")

class StaticInferenceEngine:
    """Inference engine for static single-frame ISL recognition."""

    def __init__(self, model_path: str = "models/static/static_model.keras", mapping_path: str = "models/static/class_mapping.json"):
        self.model_path = model_path
        self.mapping_path = mapping_path
        self.model: Optional[tf.keras.Model] = None
        self.mapping: Optional[ClassMappingManager] = None
        self.is_loaded = False
        self._load()

    def _load(self):
        # Fallback to model/custom_cnn.keras if models/static/static_model.keras not found
        if not os.path.exists(self.model_path) and os.path.exists("model/custom_cnn.keras"):
            self.model_path = "model/custom_cnn.keras"

        if not os.path.exists(self.model_path) or not os.path.exists(self.mapping_path):
            logger.warning(f"Static model or mapping file missing ({self.model_path}). Real-time UI will display fallback.")
            self.is_loaded = False
            return

        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.mapping = ClassMappingManager.load_json(self.mapping_path)
            self.is_loaded = True
            logger.info(f"Static inference engine loaded successfully from {self.model_path}!")
        except Exception as e:
            logger.error(f"Failed to load static inference model: {e}")
            self.is_loaded = False

    def predict(self, frame: np.ndarray, target_size: tuple = (224, 224)) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Run static prediction pipeline.
        Returns Tuple: (Predicted_Sign_Label, Confidence_Score, Top_3_List)
        """
        if not self.is_loaded or self.model is None or self.mapping is None:
            return "Waiting for trained model...", 0.0, []

        if frame is None or frame.size == 0:
            return "No Hand", 0.0, []

        # Ensure frame is in RGB format for the CNN
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame

        # Resize to model input shape (224, 224) without dividing by 255 (the Rescaling layer is internal to the model)
        resized = cv2.resize(frame_rgb, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
        tensor = resized.astype(np.float32)
        batch_tensor = np.expand_dims(tensor, axis=0)

        preds = self.model(batch_tensor, training=False).numpy()[0]
        max_idx = int(np.argmax(preds))
        confidence = float(preds[max_idx])
        label = self.mapping.get_label(max_idx)

        # Extract top 3 candidates
        top3_indices = np.argsort(preds)[::-1][:3]
        top3 = [(self.mapping.get_label(int(i)), float(preds[i])) for i in top3_indices]

        return label, confidence, top3
