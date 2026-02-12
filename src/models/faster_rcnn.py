import numpy as np
import torch
import torchvision
from typing import List, Dict, Any
from .base import BaseModel, Detection


class FasterRCNN(BaseModel):
    def __init__(self, device: str = "auto"):
        super().__init__(device)
        self.conf_threshold = 0.05

    def load_model(self, weights_path: str = None) -> None:
        if weights_path is None:
            weights = (
                torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            )
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                weights=weights
            )
        else:
            self.model = torch.load(weights_path, map_location=self.device)

        self.model.to(self.device)
        self.model.eval()

        self.model_info = {
            "name": "Faster R-CNN",
            "architecture": "ResNet50+FPN",
            "weights": weights_path,
            "device": self.device,
        }

    def predict(self, image: np.ndarray) -> List[Detection]:
        import cv2

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(image_tensor)

        detections = []
        boxes = predictions[0]["boxes"].cpu().numpy()
        scores = predictions[0]["scores"].cpu().numpy()
        labels = predictions[0]["labels"].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score >= self.conf_threshold:
                detections.append(Detection(box.tolist(), float(score), int(label)))

        return detections

    def get_model_info(self) -> Dict[str, Any]:
        if self.model is None:
            return {}

        info = self.model_info.copy()
        info["params"] = sum(p.numel() for p in self.model.parameters()) / 1e6

        return info

    def warmup(self, image_size: tuple = (640, 640)) -> None:
        dummy_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        self.predict(dummy_image)

    def set_confidence_threshold(self, threshold: float) -> None:
        self.conf_threshold = threshold
