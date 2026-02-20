"""
ONNX 模型推理支持

提供 ONNX Runtime 推理能力，支持与 PyTorch 模型的性能对比
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from .base import BaseModel, Detection

logger = logging.getLogger(__name__)


class ONNXModel(BaseModel):
    """ONNX 模型包装器"""

    def __init__(
        self,
        device: str = "auto",
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.45,
    ):
        super().__init__(device, conf_threshold)
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.names = {}
        self._is_yolo = False
        self.iou_threshold = iou_threshold
        self.model_type = None  # 'yolo', 'rtdetr', 'faster_rcnn'
        self._cpu_session = None  # 缓存 CPU session 用于回退

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

        # 检测模型类型
        self.model_type = self._detect_model_type(model_path)

        # 检测是否为 YOLO 模型（向后兼容）
        self._is_yolo = self.model_type == "yolo"

        # 尝试加载类别名称
        self.names = self._load_class_names(model_path)

        # 计算模型文件大小（MB）
        model_size_bytes = model_path.stat().st_size
        model_size_mb = model_size_bytes / (1024 * 1024)

        self.model_info = {
            "name": model_path.stem,
            "weights": str(model_path),
            "device": self.device,
            "format": "onnx",
            "input_shape": self.input_shape,
            "is_yolo": self._is_yolo,
            "model_size_mb": round(model_size_mb, 2),  # 新增：模型文件大小
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

    def _detect_model_type(self, model_path: Path) -> str:
        """
        检测 ONNX 模型类型

        Args:
            model_path: 模型文件路径

        Returns:
            模型类型: 'yolo', 'rtdetr', 'faster_rcnn'
        """
        stem = model_path.stem.lower()

        # 1. 基于文件名检测
        if "rtdetr" in stem or "rt-detr" in stem:
            return "rtdetr"
        elif "faster" in stem and "rcnn" in stem:
            return "faster_rcnn"

        # 2. 基于输出形状检测
        outputs = self.session.get_outputs()

        # Faster R-CNN: 多个输出 (boxes, labels, scores)
        if len(outputs) >= 3:
            return "faster_rcnn"

        # RT-DETR: 单个输出 [batch, 300, 84]
        if len(outputs) == 1:
            output_shape = outputs[0].shape
            if len(output_shape) == 3:
                # RT-DETR 固定 300 个查询，每个查询 84 维
                if output_shape[1] == 300 and output_shape[2] == 84:
                    return "rtdetr"

        # 默认为 YOLO
        return "yolo"

    def _get_model_input_size(self) -> Tuple[int, int]:
        """获取模型输入尺寸 (H, W)"""
        if self.input_shape and len(self.input_shape) >= 4:
            # 输入形状: [batch, channels, height, width]
            h = self.input_shape[2] if isinstance(self.input_shape[2], int) else 640
            w = self.input_shape[3] if isinstance(self.input_shape[3], int) else 640
            return (h, w)
        return (640, 640)  # 默认尺寸

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
        return yolo_index + 1  # 默认值

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
        # CoreML provider 对某些操作（如 GatherElements）支持不完整
        # 在大模型上可能导致 CoreML + CPU 混合推理不稳定
        # 策略：失败时使用 CPU provider 重试该图片
        try:
            outputs = self.session.run(
                self.output_names, {self.input_name: input_tensor}
            )
        except Exception as e:
            # 检查是否是 CoreML provider 的问题
            providers = self.session.get_providers()
            if "CoreMLExecutionProvider" in providers:
                # CoreML + CPU 混合推理失败，使用纯 CPU 重试
                logger.debug(f"CoreML 推理失败，使用 CPU 重试: {type(e).__name__}")
                outputs = self._run_with_cpu_provider(input_tensor)
            else:
                raise

        # 后处理
        detections = self._postprocess(outputs, conf, image.shape[:2])

        return detections

    def _run_with_cpu_provider(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """
        使用纯 CPU provider 执行推理（用于 CoreML 不稳定的情况）

        使用缓存的 CPU session 避免重复创建

        Args:
            input_tensor: 预处理后的输入张量

        Returns:
            ONNX 模型输出
        """
        # 如果没有缓存的 CPU session，创建一个
        if self._cpu_session is None:
            try:
                import onnxruntime as ort
            except ImportError:
                raise RuntimeError("需要安装 onnxruntime")

            model_path = self.model_info.get("weights")
            if not model_path or not Path(model_path).exists():
                raise RuntimeError("无法获取模型路径")

            self._cpu_session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            logger.debug("已创建 CPU session 缓存")

        # 使用缓存的 CPU session 执行推理
        outputs = self._cpu_session.run(
            self.output_names, {self.input_name: input_tensor}
        )

        return outputs

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

    def _apply_nms(
        self, detections: List[Detection], iou_threshold: float = None
    ) -> List[Detection]:
        """应用 NMS（非极大值抑制）到检测结果

        Args:
            detections: 检测框列表
            iou_threshold: IOU 阈值，None 时使用默认值

        Returns:
            NMS 过滤后的检测框列表
        """
        if not detections:
            return detections

        import torch

        if iou_threshold is None:
            iou_threshold = self.iou_threshold

        # 转换为 tensor
        boxes = torch.tensor([det.bbox for det in detections], dtype=torch.float32)
        scores = torch.tensor(
            [det.confidence for det in detections], dtype=torch.float32
        )

        # 应用 NMS
        try:
            from torchvision.ops import nms

            keep = nms(boxes, scores, iou_threshold)
        except ImportError:
            # 如果 torchvision 不可用，使用简化的 NMS
            keep = self._simple_nms(boxes, scores, iou_threshold)

        # 返回过滤后的检测结果
        return [detections[i] for i in keep]

    def _simple_nms(self, boxes, scores, iou_threshold: float):
        """简化的 NMS 实现（当 torchvision 不可用时使用）"""
        if len(boxes) == 0:
            return []

        # 按置信度降序排序
        idxs = torch.argsort(scores, descending=True)

        keep = []
        while len(idxs) > 0:
            # 保留最高置信度的框
            i = idxs[0]
            keep.append(i)

            # 计算当前框与剩余框的 IoU
            ious = self._compute_ious(boxes[i], boxes[idxs[1:]])

            # 过除 IoU 大于阈值的框（修复：使用 <= 而不是 <）
            idxs = idxs[1:][ious <= iou_threshold]

        return keep

    def _compute_ious(self, box, boxes):
        """计算一个框与多个框的 IoU

        Args:
            box: 单个框 [x1, y1, x2, y2]
            boxes: 多个框 [N, 4]

        Returns:
            IoU 值数组 [N]
        """
        # 计算交集
        x1 = torch.maximum(box[0], boxes[:, 0])
        y1 = torch.maximum(box[1], boxes[:, 1])
        x2 = torch.minimum(box[2], boxes[:, 2])
        y2 = torch.minimum(box[3], boxes[:, 3])

        inter_area = torch.maximum(torch.tensor(0), x2 - x1) * torch.maximum(
            torch.tensor(0), y2 - y1
        )

        # 计算并集
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - inter_area

        # 计算 IoU
        ious = inter_area / (union_area + 1e-6)  # 添加小常数避免除零

        return ious

    def _postprocess(
        self, outputs: List[np.ndarray], conf_threshold: float, original_shape: tuple
    ) -> List[Detection]:
        """
        后处理 ONNX 输出

        根据模型类型调用对应的后处理方法：
        - YOLO 系列: _postprocess_yolov8 或 _postprocess_yolov10
        - RT-DETR: _postprocess_rtdetr
        - Faster R-CNN: _postprocess_faster_rcnn

        Args:
            outputs: ONNX 模型输出
            conf_threshold: 置信度阈值
            original_shape: 原始图像尺寸 (H, W)

        Returns:
            检测框列表
        """
        if len(outputs) == 0:
            return []

        # 根据模型类型选择后处理方法
        if self.model_type == "rtdetr":
            return self._postprocess_rtdetr(outputs[0], conf_threshold, original_shape)
        elif self.model_type == "faster_rcnn":
            return self._postprocess_faster_rcnn(
                outputs, conf_threshold, original_shape
            )
        else:
            # YOLO 系列模型
            predictions = outputs[0]

            # 处理 batch 维度
            if len(predictions.shape) == 3:
                predictions = predictions[0]

            # 检查预测数量和维度
            if len(predictions.shape) < 2:
                return []

            # 检测输出格式
            shape = predictions.shape
            if shape[1] > 1000:  # YOLOv8/v9 格式 [84, 8400]
                predictions = predictions.T
                return self._postprocess_yolov8(
                    predictions, conf_threshold, original_shape
                )
            elif shape[0] > 1000:  # 已经是 [num_anchors, 84]
                return self._postprocess_yolov8(
                    predictions, conf_threshold, original_shape
                )
            else:
                # YOLOv10 格式: [num_predictions, 6]
                return self._postprocess_yolov10(
                    predictions, conf_threshold, original_shape
                )

    def _postprocess_yolov8(
        self, predictions: np.ndarray, conf_threshold: float, original_shape: tuple
    ) -> List[Detection]:
        """
        处理 YOLOv8/v9 格式的 ONNX 输出
        格式: [num_anchors, 84]
        其中 84 = 4 (bbox: x, y, w, h) + 80 (class scores)
        """
        detections = []

        if len(predictions.shape) != 2:
            return detections

        num_anchors = predictions.shape[0]
        num_values = predictions.shape[1]

        if num_values < 5:
            return detections

        # 获取模型输入尺寸
        model_h, model_w = self._get_model_input_size()

        for i in range(num_anchors):
            pred = predictions[i]

            # YOLOv8/v9: [x_center, y_center, w, h, class0_score, class1_score, ...]
            x_center, y_center, w, h = pred[0], pred[1], pred[2], pred[3]

            # 计算类别分数
            if num_values > 4:
                class_scores = pred[4:]
                class_id = int(np.argmax(class_scores))
                conf = float(class_scores[class_id])
            else:
                class_id = 0
                conf = 0.0

            # 置信度过滤
            if conf < conf_threshold:
                continue

            # 转换为 xyxy 格式
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2

            # 限制在模型输出尺寸内
            x1 = max(0, min(model_w, x1))
            y1 = max(0, min(model_h, y1))
            x2 = max(0, min(model_w, x2))
            y2 = max(0, min(model_h, y2))

            # 过滤太小的框
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue

            # 缩放到原图尺寸
            orig_h, orig_w = original_shape[:2]
            scale_x = orig_w / model_w
            scale_y = orig_h / model_h

            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y

            # 限制在原图尺寸内
            x1_scaled = max(0, min(orig_w, x1_scaled))
            y1_scaled = max(0, min(orig_h, y1_scaled))
            x2_scaled = max(0, min(orig_w, x2_scaled))
            y2_scaled = max(0, min(orig_h, y2_scaled))

            # 确保有效的框
            if x2_scaled > x1_scaled and y2_scaled > y1_scaled:
                # 将 YOLO 类别索引（0-79）映射到 COCO 类别 ID（1-90，不连续）
                coco_class_id = self._yolo_index_to_coco_id(int(class_id))
                detections.append(
                    Detection(
                        bbox=[
                            float(x1_scaled),
                            float(y1_scaled),
                            float(x2_scaled),
                            float(y2_scaled),
                        ],
                        confidence=float(conf),
                        class_id=coco_class_id,
                    )
                )

        # 应用 NMS
        detections = self._apply_nms(detections)

        return detections

    def _postprocess_yolov10(
        self, predictions: np.ndarray, conf_threshold: float, original_shape: tuple
    ) -> List[Detection]:
        """
        处理 YOLOv10 格式的 ONNX 输出
        格式: [num_predictions, 6]
        其中 6 = [x1, y1, x2, y2, conf, class_id]
        """
        detections = []

        if len(predictions.shape) != 2:
            return detections

        num_preds = predictions.shape[0]
        num_values = predictions.shape[1]

        # YOLOv10 格式需要至少 6 个值
        if num_values < 6:
            return detections

        # 获取模型输入尺寸
        model_h, model_w = self._get_model_input_size()

        for i in range(num_preds):
            pred = predictions[i]

            # YOLOv10: [x1, y1, x2, y2, conf, class_id]
            x1, y1, x2, y2, conf, class_id = (
                pred[0],
                pred[1],
                pred[2],
                pred[3],
                pred[4],
                pred[5],
            )

            # 置信度过滤
            if conf < conf_threshold:
                continue

            # 限制在模型输出尺寸内
            x1 = max(0, min(model_w, x1))
            y1 = max(0, min(model_h, y1))
            x2 = max(0, min(model_w, x2))
            y2 = max(0, min(model_h, y2))

            # 过滤太小的框
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue

            # 缩放到原图尺寸
            orig_h, orig_w = original_shape[:2]
            scale_x = orig_w / model_w
            scale_y = orig_h / model_h

            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y

            # 限制在原图尺寸内
            x1_scaled = max(0, min(orig_w, x1_scaled))
            y1_scaled = max(0, min(orig_h, y1_scaled))
            x2_scaled = max(0, min(orig_w, x2_scaled))
            y2_scaled = max(0, min(orig_h, y2_scaled))

            # 确保有效的框
            if x2_scaled > x1_scaled and y2_scaled > y1_scaled:
                # 将 YOLO 类别索引（0-79）映射到 COCO 类别 ID（1-90，不连续）
                coco_class_id = self._yolo_index_to_coco_id(int(class_id))
                detections.append(
                    Detection(
                        bbox=[
                            float(x1_scaled),
                            float(y1_scaled),
                            float(x2_scaled),
                            float(y2_scaled),
                        ],
                        confidence=float(conf),
                        class_id=coco_class_id,
                    )
                )

        # 应用 NMS
        detections = self._apply_nms(detections)

        return detections

    def _postprocess_rtdetr(
        self, output: np.ndarray, conf_threshold: float, original_shape: tuple
    ) -> List[Detection]:
        """
        处理 RT-DETR 格式的 ONNX 输出
        格式: [batch, num_queries, 84]
        其中 84 = 4 (bbox) + 80 (class scores)

        Args:
            output: RT-DETR 输出 [1, 300, 84]
            conf_threshold: 置信度阈值
            original_shape: 原始图像尺寸 (H, W)

        Returns:
            检测框列表
        """
        detections = []

        # 处理 batch 维度
        if len(output.shape) == 3:
            output = output[0]  # [300, 84]

        if len(output.shape) != 2:
            return detections

        num_queries = output.shape[0]
        num_values = output.shape[1]

        if num_values < 84:
            return detections

        # 获取模型输入尺寸
        model_h, model_w = self._get_model_input_size()

        for i in range(num_queries):
            pred = output[i]

            # RT-DETR: [cx, cy, w, h (归一化), class0_score, ..., class79_score]
            # bbox 是归一化的 cxcywh 格式，范围 0-1
            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
            class_scores = pred[4:84]

            # 找到最高分类分数
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            # 置信度过滤
            if confidence < conf_threshold:
                continue

            # 转换归一化的 cxcywh 为绝对坐标的 xyxy
            # cx, cy, w, h 都在 0-1 范围内
            x1 = (cx - w / 2) * model_w
            y1 = (cy - h / 2) * model_h
            x2 = (cx + w / 2) * model_w
            y2 = (cy + h / 2) * model_h

            # 限制在模型输出尺寸内
            x1 = max(0, min(model_w, x1))
            y1 = max(0, min(model_h, y1))
            x2 = max(0, min(model_w, x2))
            y2 = max(0, min(model_h, y2))

            # 过滤太小的框
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue

            # 缩放到原图尺寸
            orig_h, orig_w = original_shape[:2]
            scale_x = orig_w / model_w
            scale_y = orig_h / model_h

            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y

            # 限制在原图尺寸内
            x1_scaled = max(0, min(orig_w, x1_scaled))
            y1_scaled = max(0, min(orig_h, y1_scaled))
            x2_scaled = max(0, min(orig_w, x2_scaled))
            y2_scaled = max(0, min(orig_h, y2_scaled))

            # 确保有效的框
            if x2_scaled > x1_scaled and y2_scaled > y1_scaled:
                # 将 YOLO 类别索引（0-79）映射到 COCO 类别 ID（1-90，不连续）
                coco_class_id = self._yolo_index_to_coco_id(class_id)
                detections.append(
                    Detection(
                        bbox=[
                            float(x1_scaled),
                            float(y1_scaled),
                            float(x2_scaled),
                            float(y2_scaled),
                        ],
                        confidence=confidence,
                        class_id=coco_class_id,
                    )
                )

        # RT-DETR 已经包含 NMS，不需要额外应用
        return detections

    def _postprocess_faster_rcnn(
        self, outputs: List[np.ndarray], conf_threshold: float, original_shape: tuple
    ) -> List[Detection]:
        """
        处理 Faster R-CNN 格式的 ONNX 输出
        格式: 多个输出 [boxes, labels, scores]

        Args:
            outputs: Faster R-CNN 输出列表
            conf_threshold: 置信度阈值
            original_shape: 原始图像尺寸 (H, W)

        Returns:
            检测框列表
        """
        detections = []

        if len(outputs) < 3:
            return detections

        # Faster R-CNN 输出三个张量
        # outputs[0]: boxes [N, 4]
        # outputs[1]: labels [N]
        # outputs[2]: scores [N]
        boxes = outputs[0]
        labels = outputs[1]
        scores = outputs[2]

        # 处理 batch 维度
        if len(boxes.shape) == 3:
            boxes = boxes[0]
            labels = labels[0]
            scores = scores[0]

        # 获取模型输入尺寸
        model_h, model_w = self._get_model_input_size()

        # 遍历所有检测
        for box, label, score in zip(boxes, labels, scores):
            # 置信度过滤
            if score < conf_threshold:
                continue

            # 解析 bbox (xyxy 格式)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # 限制在模型输出尺寸内
            x1 = max(0, min(model_w, x1))
            y1 = max(0, min(model_h, y1))
            x2 = max(0, min(model_w, x2))
            y2 = max(0, min(model_h, y2))

            # 过滤太小的框
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue

            # 缩放到原图尺寸
            orig_h, orig_w = original_shape[:2]
            scale_x = orig_w / model_w
            scale_y = orig_h / model_h

            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y

            # 限制在原图尺寸内
            x1_scaled = max(0, min(orig_w, x1_scaled))
            y1_scaled = max(0, min(orig_h, y1_scaled))
            x2_scaled = max(0, min(orig_w, x2_scaled))
            y2_scaled = max(0, min(orig_h, y2_scaled))

            # 确保有效的框
            if x2_scaled > x1_scaled and y2_scaled > y1_scaled:
                # Faster R-CNN 的 label 已经是 COCO ID，不需要映射
                detections.append(
                    Detection(
                        bbox=[
                            float(x1_scaled),
                            float(y1_scaled),
                            float(x2_scaled),
                            float(y2_scaled),
                        ],
                        confidence=float(score),
                        class_id=int(label),
                    )
                )

        # Faster R-CNN 已经包含 NMS，不需要额外应用
        return detections

        # 处理 batch 维度
        if len(predictions.shape) == 3:
            predictions = predictions[0]

        # 检查预测数量和维度
        if len(predictions.shape) < 2:
            return detections

        num_preds = predictions.shape[0]
        num_values = predictions.shape[1]

        # YOLO 格式需要至少 6 个值
        if num_values < 6:
            return detections

        # 处理每个预测
        for i in range(num_preds):
            pred = predictions[i]

            # 提取值
            x, y, w_or_x2, h_or_y2, conf, class_id = pred[:6]

            # 置信度过滤
            if conf < conf_threshold:
                continue

            # 根据 YOLO 版本解析坐标
            # 方式 1: [x, y, w, h, conf, class] - 中心点 + 宽高
            # 方式 2: [x1, y1, x2, y2, conf, class] - 左上角 + 右下角

            # 检测格式：通过 w/h 是否为负数或超过图像尺寸来判断
            # 对于角点格式：x1, y1, x2, y2 应该满足 0 <= x1 < x2 <= 640
            # 对于中心点格式：x, y, w, h 应该满足 0 <= x < 640, w > 0

            model_h, model_w = 640, 640

            # 尝试两种格式，选择合理的一种
            # 格式 1: 中心点 [x, y, w, h]
            x1_center = x - w_or_x2 / 2
            y1_center = y - h_or_y2 / 2
            x2_center = x + w_or_x2 / 2
            y2_center = y + h_or_y2 / 2

            # 格式 2: 角点 [x1, y1, x2, y2]
            x1_corner = x
            y1_corner = y
            x2_corner = w_or_x2
            y2_corner = h_or_y2

            # 判断哪种格式更合理
            # 格式 1 检查：中心点应该在图像内，宽高应该为正
            center_format_valid = (
                0 <= x < model_w
                and 0 <= y < model_h
                and w_or_x2 > 0
                and h_or_y2 > 0
                and x1_center >= 0
                and y1_center >= 0
            )

            # 格式 2 检查：x1 < x2, y1 < y2，且在合理范围内
            corner_format_valid = (
                x < x2
                and y < y2
                and x2 <= model_w * 1.1  # 允许稍微超出
                and y2 <= model_h * 1.1
            )

            if center_format_valid:
                # 使用中心点格式
                x1, y1, x2, y2 = x1_center, y1_center, x2_center, y2_center
            elif corner_format_valid:
                # 使用角点格式
                x1, y1, x2, y2 = x1_corner, y1_corner, x2_corner, y2_corner
            else:
                # 都不合理，跳过
                continue

            # 限制在模型输出尺寸内
            x1 = max(0, min(model_w, x1))
            y1 = max(0, min(model_h, y1))
            x2 = max(0, min(model_w, x2))
            y2 = max(0, min(model_h, y2))

            # 过滤太小的框（可能是误检）
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue

            # 缩放到原图尺寸
            orig_h, orig_w = original_shape[:2]
            scale_x = orig_w / model_w
            scale_y = orig_h / model_h

            x1_scaled = x1 * scale_x
            y1_scaled = y1 * scale_y
            x2_scaled = x2 * scale_x
            y2_scaled = y2 * scale_y

            # 限制在原图尺寸内
            x1_scaled = max(0, min(orig_w, x1_scaled))
            y1_scaled = max(0, min(orig_h, y1_scaled))
            x2_scaled = max(0, min(orig_w, x2_scaled))
            y2_scaled = max(0, min(orig_h, y2_scaled))

            # 确保有效的框
            if x2_scaled > x1_scaled and y2_scaled > y1_scaled:
                detections.append(
                    Detection(
                        bbox=[
                            float(x1_scaled),
                            float(y1_scaled),
                            float(x2_scaled),
                            float(y2_scaled),
                        ],
                        confidence=float(conf),
                        class_id=int(class_id),
                    )
                )

        return detections

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.session is None:
            return {}

        info = self.model_info.copy()

        # 估算参数量：使用模型文件大小 / 4 作为粗略估算
        # 因为 fp32 模型每个参数占 4 字节
        # 注意：这只是一个粗略估算，ONNX 模型可能包含优化/压缩
        if "model_size_mb" in info and "params" not in info:
            # MB ≈ Million params（fp32 精度）
            estimated_params = round(info["model_size_mb"], 2)
            info["params"] = estimated_params

        # 确保 model_size_mb 字段存在
        if "model_size_mb" not in info and "weights" in info:
            weights_path = Path(info["weights"])
            if weights_path.exists():
                model_size_bytes = weights_path.stat().st_size
                info["model_size_mb"] = round(model_size_bytes / (1024 * 1024), 2)

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
