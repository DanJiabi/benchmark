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

    def _yolo_index_to_coco_id(self, yolo_index: int) -> int:
        """将 YOLO 类别索引（0-79）映射到 COCO 类别 ID（1-90，不连续）"""
        coco_id_mapping = [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            27,
            28,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            67,
            70,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
        ]
        if 0 <= yolo_index < len(coco_id_mapping):
            return coco_id_mapping[yolo_index]
        return yolo_index + 1  # 默认值（向后兼容）

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
                # 将 YOLO 类别索引（0-79）映射到 COCO 类别 ID（1-90，不连续）
                coco_class_id = self._yolo_index_to_coco_id(cls)
                detections.append(Detection(xyxy, conf, coco_class_id))

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
