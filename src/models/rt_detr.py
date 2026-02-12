import numpy as np
import torch
from typing import List, Dict, Any
from .base import BaseModel, Detection


class RTDETR(BaseModel):
    def __init__(self, device: str = "auto"):
        super().__init__(device)

    def load_model(self, weights_path: str) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights_path)
        self.model.to(self.device)

        self.model_info = {
            "name": "RT-DETR",
            "weights": weights_path,
            "device": self.device,
        }

    def predict(self, image: np.ndarray) -> List[Detection]:
        results = self.model(image, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy.cpu().numpy()[0].tolist()
                conf = float(box.conf.cpu().numpy()[0])
                cls = int(box.cls.cpu().numpy()[0])
                detections.append(Detection(xyxy, conf, cls))

        return detections

    def get_model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {}

        info = self.model_info.copy()
        info["params"] = sum(p.numel() for p in self.model.model.parameters()) / 1e6

        return info

    def warmup(self, image_size: tuple = (640, 640)) -> None:
        dummy_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        self.predict(dummy_image)
