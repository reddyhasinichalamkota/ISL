import os
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Union, List
from src.preprocessing.video_preprocessing import preprocess_video_sequence
from src.utils.class_mapping import ClassMappingManager
from src.utils.logger import get_logger

logger = get_logger("DynamicInferenceEngine")

class DynamicInferenceEngine:
    """Inference engine for dynamic sequence ISL recognition."""

    def __init__(self, model_path: str = "models/dynamic/dynamic_model.keras", mapping_path: str = "models/dynamic/class_mapping.json"):
        self.model_path = model_path
        self.mapping_path = mapping_path
        self.model: Optional[tf.keras.Model] = None
        self.mapping: Optional[ClassMappingManager] = None
        self.is_loaded = False
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.mapping_path):
            logger.warning(f"Dynamic model or mapping file missing ({self.model_path}). Real-time UI will display fallback.")
            self.is_loaded = False
            return

        try:
            self.model = tf.keras.models.load_model(self.model_path)
            self.mapping = ClassMappingManager.load_json(self.mapping_path)
            self.is_loaded = True
            logger.info("Dynamic inference engine loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load dynamic inference model: {e}")
            self.is_loaded = False

    def predict_sequence(
        self,
        sequence_input: Union[np.ndarray, List[np.ndarray], str],
        sequence_length: int = 30,
        target_size: tuple = (224, 224)
    ) -> Tuple[str, float]:
        """
        Run dynamic sequence prediction pipeline.
        Returns Tuple: (Predicted_Sign_Label, Confidence_Score)
        """
        if not self.is_loaded or self.model is None or self.mapping is None:
            return "Waiting for trained model...", 0.0

        seq_tensor = preprocess_video_sequence(
            sequence_input,
            target_len=sequence_length,
            target_size=target_size,
            normalize=False
        ).astype(np.float32)
        batch_seq = np.expand_dims(seq_tensor, axis=0)

        preds = self.model.predict(batch_seq, verbose=0)[0]
        max_idx = int(np.argmax(preds))
        confidence = float(preds[max_idx])
        label = self.mapping.get_label(max_idx)

        return label, confidence
