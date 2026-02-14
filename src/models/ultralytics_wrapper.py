import numpy as np
from typing import List, Dict, Any
from .base import BaseModel, Detection


class UltralyticsWrapper(BaseModel):
    def __init__(self, device: str = "auto", conf_threshold: float = 0.001):
        super().__init__(device, conf_threshold)
        self.model_type = None
        self.names = {}

    def load_model(self, weights_path: str, model_name: str = "Ultralytics") -> None:
        from ultralytics import YOLO

        self.model_type = model_name
        self.model = YOLO(weights_path)
        self.model.to(self.device)

        self.names = self.model.names if hasattr(self.model, "names") else {}

        self.model_info = {
            "name": model_name,
            "weights": weights_path,
            "device": self.device,
        }

    def predict(
        self, image: np.ndarray, conf_threshold: float = None
    ) -> List[Detection]:
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        results = self.model(image, verbose=False, conf=conf)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy()[0].tolist()
                conf = float(box.conf.cpu().numpy()[0])
                cls = int(box.cls.cpu().numpy()[0])
                detections.append(Detection(xyxy, conf, cls + 1))

        return detections

    def get_model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {}

        info = self.model_info.copy()
        info["params"] = sum(p.numel() for p in self.model.model.parameters()) / 1e6

        model_yaml = self.model.model.yaml if hasattr(self.model.model, "yaml") else {}
        if model_yaml:
            info["model_yaml"] = model_yaml

        return info

    def warmup(self, image_size: tuple = (640, 640)) -> None:
        dummy_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        self.predict(dummy_image)
