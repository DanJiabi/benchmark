from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np


class Detection:
    def __init__(self, bbox: List[float], confidence: float, class_id: int):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id


class BaseModel(ABC):
    def __init__(self, device: str = "auto", conf_threshold: float = 0.001):
        self.device = self._get_device(device)
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_info = {}

    @abstractmethod
    def load_model(self, weights_path: str) -> None:
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> List[Detection]:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def warmup(self, image_size: tuple = (640, 640)) -> None:
        pass

    def _get_device(self, device: str) -> str:
        if device != "auto":
            return device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
