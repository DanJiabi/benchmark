"""模型基类和数据结构定义.

提供所有目标检测模型的统一接口和基础功能。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os
import numpy as np


class Detection:
    """目标检测结果.

    Attributes:
        bbox: 边界框坐标 [x1, y1, x2, y2]
        confidence: 检测置信度 (0-1)
        class_id: 类别 ID (COCO 格式)

    Example:
        >>> det = Detection([100, 200, 300, 400], 0.95, 1)
        >>> print(det.bbox)
        [100, 200, 300, 400]
    """

    def __init__(self, bbox: List[float], confidence: float, class_id: int):
        """初始化检测结果.

        Args:
            bbox: 边界框坐标 [x1, y1, x2, y2]
            confidence: 检测置信度
            class_id: 类别 ID
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id


class BaseModel(ABC):
    """目标检测模型基类.

    所有模型实现必须继承此类并实现以下方法:
    - load_model(): 加载模型权重
    - predict(): 执行推理
    - get_model_info(): 获取模型信息
    - warmup(): 模型预热

    Attributes:
        device: 推理设备 ('cuda', 'mps', 'cpu', 'auto')
        conf_threshold: 置信度阈值
        model: 模型实例
        model_info: 模型信息字典

    Example:
        >>> class MyModel(BaseModel):
        ...     def load_model(self, weights_path):
        ...         self.model = load_my_model(weights_path)
        ...     def predict(self, image):
        ...         return self.model(image)
    """

    def __init__(self, device: str = "auto", conf_threshold: float = 0.001):
        """初始化模型.

        Args:
            device: 推理设备
                - 'auto': 自动检测（cuda > mps > cpu）
                - 'cuda': NVIDIA GPU
                - 'mps': Apple Silicon GPU
                - 'cpu': CPU
            conf_threshold: 置信度阈值，默认 0.001
        """
        self.device = self._get_device(device)
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_info = {}

    @abstractmethod
    def load_model(self, weights_path: str) -> None:
        """加载模型权重.

        Args:
            weights_path: 权重文件路径
        """
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> List[Detection]:
        """执行目标检测推理.

        Args:
            image: 输入图像 (BGR, numpy array)

        Returns:
            检测结果列表
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息.

        Returns:
            包含模型名称、参数量、输入尺寸等信息的字典
        """
        pass

    @abstractmethod
    def warmup(self, image_size: tuple = (640, 640)) -> None:
        """模型预热.

        Args:
            image_size: 预热图像尺寸 (H, W)
        """
        pass

    def _get_device(self, device: str) -> str:
        """获取推理设备.

        Args:
            device: 设备字符串

        Returns:
            实际使用的设备字符串
        """
        import torch

        selected_device = device
        fallback_reason = None

        if device != "auto":
            # 用户明确指定了设备
            if device == "mps":
                # 检查 MPS 是否可用
                if (
                    not hasattr(torch.backends, "mps")
                    or not torch.backends.mps.is_available()
                ):
                    fallback_reason = "MPS 不可用（非 Apple Silicon 设备）"
                    selected_device = "cpu"
                elif os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
                    fallback_reason = "PYTORCH_ENABLE_MPS_FALLBACK 未设置"
                    selected_device = "cpu"
        else:
            # 自动选择设备
            if torch.cuda.is_available():
                selected_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # 检查环境变量
                if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
                    selected_device = "mps"
                else:
                    fallback_reason = (
                        "PYTORCH_ENABLE_MPS_FALLBACK 未设置，使用 CPU 以确保兼容性"
                    )
                    selected_device = "cpu"
            else:
                selected_device = "cpu"

        # 打印设备信息
        self._print_device_info(device, selected_device, fallback_reason)

        return selected_device

    def _print_device_info(
        self,
        requested_device: str,
        actual_device: str,
        fallback_reason: Optional[str] = None,
    ):
        """打印设备选择信息.

        Args:
            requested_device: 用户请求的设备
            actual_device: 实际使用的设备
            fallback_reason: 回退原因（如果有）
        """
        print(f"\n🖥️  推理设备信息:")
        print(f"   请求设备: {requested_device}")
        print(f"   实际设备: {actual_device}")

        if fallback_reason:
            print(f"   ⚠️  设备回退原因: {fallback_reason}")
            if actual_device == "cpu" and "MPS" in fallback_reason:
                print(f"   💡 提示: Apple Silicon 的 MPS 需要设置环境变量才能使用")
                print(f"      运行: export PYTORCH_ENABLE_MPS_FALLBACK=1")
                print(f"      然后: od-benchmark --device mps ...")

        if actual_device == "cuda":
            import torch

            print(f"   ✅ 使用 NVIDIA GPU 加速")
        elif actual_device == "mps":
            print(f"   ✅ 使用 Apple Silicon GPU 加速")
            print(f"   ⚠️  注意: 部分操作可能回退到 CPU")
        else:
            print(f"   📝 使用 CPU 推理")
        print()
