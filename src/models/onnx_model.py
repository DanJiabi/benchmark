"""
ONNX 模型推理支持

提供 ONNX Runtime 推理能力，支持与 PyTorch 模型的性能对比
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from .base import BaseModel, Detection


class ONNXModel(BaseModel):
    """ONNX 模型包装器"""

    def __init__(self, device: str = "auto", conf_threshold: float = 0.001):
        super().__init__(device, conf_threshold)
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.names = {}
        self._is_yolo = False

    def load_model(self, model_path: str) -> None:
        """
        加载 ONNX 模型

        Args:
            model_path: ONNX 模型文件路径 (.onnx)
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "需要安装 onnxruntime 以使用 ONNX 推理。\n"
                "运行: pip install onnxruntime\n"
                "或: pip install onnxruntime-gpu (GPU版本)"
            )

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX 模型文件不存在: {model_path}")

        # 设置推理提供程序
        providers = self._get_providers()

        # 创建推理会话
        self.session = ort.InferenceSession(str(model_path), providers=providers)

        # 获取输入信息
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.input_shape = input_meta.shape

        # 获取输出信息
        self.output_names = [o.name for o in self.session.get_outputs()]

        # 检测是否为 YOLO 模型
        self._is_yolo = "yolo" in model_path.stem.lower()

        # 尝试加载类别名称
        self.names = self._load_class_names(model_path)

        self.model_info = {
            "name": model_path.stem,
            "weights": str(model_path),
            "device": self.device,
            "format": "onnx",
            "input_shape": self.input_shape,
            "is_yolo": self._is_yolo,
        }

    def _get_providers(self) -> List[str]:
        """获取可用的推理提供程序"""
        try:
            import onnxruntime as ort
        except ImportError:
            return ["CPUExecutionProvider"]

        available = ort.get_available_providers()

        # MPS 设备使用 CoreML（Apple Silicon）
        if self.device == "mps" or self.device == "auto":
            if "CoreMLExecutionProvider" in available:
                return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        # CUDA 设备
        if self.device == "cuda":
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Auto 模式：尝试 CoreML 或 CUDA
        if self.device == "auto":
            if "CoreMLExecutionProvider" in available:
                return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
            elif "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # 默认使用 CPU
        return ["CPUExecutionProvider"]

    def _load_class_names(self, model_path: Path) -> Dict[int, str]:
        """尝试从 ONNX 模型加载类别名称"""
        # 尝试从同名 .yaml 或 .json 文件加载
        yaml_path = model_path.with_suffix(".yaml")
        json_path = model_path.with_suffix(".json")

        if yaml_path.exists():
            try:
                import yaml

                with open(yaml_path, "r") as f:
                    data = yaml.safe_load(f)
                    if "names" in data:
                        return {int(k): v for k, v in data["names"].items()}
            except Exception:
                pass

        if json_path.exists():
            try:
                import json

                with open(json_path, "r") as f:
                    data = json.load(f)
                    if "names" in data:
                        return {int(k): v for k, v in data["names"].items()}
            except Exception:
                pass

        # 返回默认 COCO 类别
        return {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush",
        }

    def predict(
        self, image: np.ndarray, conf_threshold: Optional[float] = None
    ) -> List[Detection]:
        """
        执行 ONNX 推理

        Args:
            image: 输入图像 (BGR, numpy array)
            conf_threshold: 置信度阈值

        Returns:
            检测框列表
        """
        if self.session is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        # 预处理
        input_tensor = self._preprocess(image)

        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # 后处理
        detections = self._postprocess(outputs, conf, image.shape[:2])

        return detections

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理

        Args:
            image: BGR 图像

        Returns:
            预处理后的输入张量
        """
        # 获取输入尺寸
        input_h, input_w = 640, 640  # 默认尺寸
        if self.input_shape:
            # 处理动态维度
            shape = list(self.input_shape)
            if len(shape) == 4:
                if isinstance(shape[2], int) and shape[2] > 0:
                    input_h = shape[2]
                if isinstance(shape[3], int) and shape[3] > 0:
                    input_w = shape[3]

        # 调整图像尺寸
        img = cv2.resize(image, (input_w, input_h))

        # BGR to RGB
        img = img[:, :, ::-1]

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        # 归一化到 0-1
        img = img.astype(np.float32) / 255.0

        # 添加 batch 维度
        img = np.expand_dims(img, axis=0)

        return img

    def _postprocess(
        self, outputs: List[np.ndarray], conf_threshold: float, original_shape: tuple
    ) -> List[Detection]:
        """
        后处理 ONNX 输出

        注意：这是一个简化的后处理，适用于性能测试。
        实际生产环境应使用完整的 NMS 后处理。

        Args:
            outputs: ONNX 模型输出
            conf_threshold: 置信度阈值
            original_shape: 原始图像尺寸 (H, W)

        Returns:
            检测框列表
        """
        detections = []

        # YOLOv8 ONNX 输出格式: [batch, num_predictions, 84] (80 classes + 4 bbox)
        # 简化处理：返回一些虚拟检测框用于性能测试
        # 在实际场景中，应该使用 ultralytics 的 NMS 或 onnxruntime 的 NMS

        if len(outputs) > 0:
            predictions = outputs[0]

            # 处理输出
            if len(predictions.shape) == 3:
                predictions = predictions[0]  # 移除 batch 维度

            # 查找高置信度的检测
            # 对于 YOLO 格式 [num_predictions, 84]
            # 84 = x, y, w, h + 80 classes
            if predictions.shape[1] >= 5:
                for i in range(min(10, predictions.shape[0])):  # 最多取10个
                    pred = predictions[i]

                    # 提取边界框 (假设前4个是 x, y, w, h)
                    x, y, w, h = pred[0], pred[1], pred[2], pred[3]

                    # 提取类别置信度
                    if predictions.shape[1] > 4:
                        class_scores = pred[4:]
                        class_id = int(np.argmax(class_scores))
                        conf = float(class_scores[class_id])
                    else:
                        class_id = 0
                        conf = 0.5

                    # 只保留高置信度的检测
                    if conf >= conf_threshold:
                        # 转换为 xyxy 格式并缩放到原图尺寸
                        scale_x = original_shape[1] / 640.0
                        scale_y = original_shape[0] / 640.0

                        x1 = max(0, (x - w / 2) * scale_x)
                        y1 = max(0, (y - h / 2) * scale_y)
                        x2 = min(original_shape[1], (x + w / 2) * scale_x)
                        y2 = min(original_shape[0], (y + h / 2) * scale_y)

                        if x2 > x1 and y2 > y1:
                            detection = Detection(
                                bbox=[float(x1), float(y1), float(x2), float(y2)],
                                confidence=conf,
                                class_id=class_id,
                            )
                            detections.append(detection)

        return detections

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.session is None:
            return {}

        info = self.model_info.copy()

        # 获取输入/输出信息
        info["inputs"] = [
            {"name": i.name, "shape": i.shape, "type": i.type}
            for i in self.session.get_inputs()
        ]

        info["outputs"] = [
            {"name": o.name, "shape": o.shape, "type": o.type}
            for o in self.session.get_outputs()
        ]

        # 获取使用的执行提供程序
        info["providers"] = self.session.get_providers()

        return info

    def warmup(self, image_size: tuple = (640, 640)) -> None:
        """模型预热"""
        if self.session is None:
            return

        # 创建虚拟输入
        dummy_image = np.random.randint(
            0, 255, (image_size[0], image_size[1], 3), dtype=np.uint8
        )

        # 执行一次推理
        try:
            self.predict(dummy_image)
        except Exception as e:
            # 预热失败不影响正常使用
            pass


# 在文件开头添加 cv2 导入
try:
    import cv2
except ImportError:
    # 如果没有 cv2，使用一个虚拟函数
    class DummyCV2:
        @staticmethod
        def resize(img, size):
            # 简单的 numpy resize
            from PIL import Image

            return np.array(Image.fromarray(img).resize(size))

    cv2 = DummyCV2()
