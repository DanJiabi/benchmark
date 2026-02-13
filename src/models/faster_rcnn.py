import numpy as np
import torch
import torchvision
from typing import List, Dict, Any
from .base import BaseModel, Detection


class FasterRCNN(BaseModel):
    def __init__(self, device: str = "auto", conf_threshold: float = 0.05):
        super().__init__(device, conf_threshold)
        self.conf_threshold = conf_threshold

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

        self.names = {
            1: "person",
            2: "bicycle",
            3: "car",
            4: "motorcycle",
            5: "airplane",
            6: "bus",
            7: "train",
            8: "truck",
            9: "boat",
            10: "traffic light",
            11: "fire hydrant",
            13: "stop sign",
            14: "parking meter",
            15: "bench",
            16: "bird",
            17: "cat",
            18: "dog",
            19: "horse",
            20: "sheep",
            21: "cow",
            22: "elephant",
            23: "bear",
            24: "zebra",
            25: "giraffe",
            27: "backpack",
            28: "umbrella",
            31: "handbag",
            32: "tie",
            33: "suitcase",
            34: "frisbee",
            35: "skis",
            36: "snowboard",
            37: "sports ball",
            38: "kite",
            39: "baseball bat",
            40: "baseball glove",
            41: "skateboard",
            42: "surfboard",
            43: "tennis racket",
            44: "bottle",
            46: "wine glass",
            47: "cup",
            48: "fork",
            49: "knife",
            50: "spoon",
            51: "bowl",
            52: "banana",
            53: "apple",
            54: "sandwich",
            55: "orange",
            56: "broccoli",
            57: "carrot",
            58: "hot dog",
            59: "pizza",
            60: "donut",
            61: "cake",
            62: "chair",
            63: "couch",
            64: "potted plant",
            65: "bed",
            67: "dining table",
            70: "toilet",
            72: "tv",
            73: "laptop",
            74: "mouse",
            75: "remote",
            76: "keyboard",
            77: "cell phone",
            78: "microwave",
            79: "oven",
            80: "toaster",
            81: "sink",
            82: "refrigerator",
            84: "book",
            85: "clock",
            86: "vase",
            87: "scissors",
            88: "teddy bear",
            89: "hair drier",
            90: "toothbrush",
        }

        self.model_info = {
            "name": "Faster R-CNN",
            "architecture": "ResNet50+FPN",
            "weights": weights_path,
            "device": self.device,
        }

    def predict(
        self, image: np.ndarray, conf_threshold: float = None
    ) -> List[Detection]:
        import cv2

        threshold = (
            conf_threshold if conf_threshold is not None else self.conf_threshold
        )

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
            if score >= threshold:
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
