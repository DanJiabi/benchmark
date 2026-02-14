"""
模拟模型 - 用于调试和演示

这个模块提供了一个包装器，可以模拟自定义模型的行为。
在实际使用中，用户应该替换为真实的自定义模型实现。
"""

from typing import List, Dict, Any, Optional
import numpy as np

from .base import BaseModel, Detection
from .ultralytics_wrapper import UltralyticsWrapper


class SimulatedModel(BaseModel):
    """
    模拟用户自定义模型

    在开发和调试阶段，这个类使用 YOLOv8n 作为基础，
    但模拟成一个不同的用户模型，用于测试对比分析功能。

    实际使用时，用户应该创建真实的自定义模型类。
    """

    def __init__(self, device: str = "auto", conf_threshold: float = 0.001):
        super().__init__(device, conf_threshold)

        # 内部使用实际的模型（这里使用 YOLOv8n）
        self._inner_model = None
        self._baseline_model = None
        self.simulation_mode = "baseline"  # 或 "custom"

    def load_model(
        self, weights_path: Optional[str] = None, model_name: str = "SimulatedModel"
    ) -> None:
        """
        加载模型

        Args:
            weights_path: 权重文件路径（如果 None，使用模拟模式）
            model_name: 模型名称（用于标识）
        """
        from . import create_model

        # 加载基础模型（YOLOv8n）
        if weights_path:
            self._inner_model = create_model(
                "yolov8n", device=self.device, conf_threshold=self.conf_threshold
            )
            from . import load_model_wrapper

            load_model_wrapper(self._inner_model, weights_path, model_name)
        else:
            # 模拟模式：加载两次相同的模型
            self._inner_model = create_model(
                "yolov8n", device=self.device, conf_threshold=self.conf_threshold
            )
            from . import load_model_wrapper

            load_model_wrapper(self._inner_model, "models_cache/yolov8n.pt", "yolov8n")

        self.model_info = {
            "name": model_name,
            "weights": weights_path,
            "device": self.device,
            "simulated": True,
        }

    def predict(
        self, image: np.ndarray, conf_threshold: Optional[float] = None
    ) -> List[Detection]:
        """
        执行推理

        在模拟模式下，可以添加一些变化来模拟不同的模型行为：
        - 调整置信度
        - 添加延迟（模拟较慢的模型）
        - 过滤部分检测（模拟较准确的模型）
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        # 获取基础模型的预测
        detections = self._inner_model.predict(image, conf)

        # 根据模拟模式调整检测
        if self.simulation_mode == "custom":
            # 模拟自定义模型的特性
            # 示例 1: 降低置信度（模拟不够准确）
            detections = [
                Detection(
                    bbox=det.bbox,
                    confidence=det.confidence * 0.85,  # 降低 15% 置信度
                    class_id=det.class_id,
                )
                for det in detections
            ]

            # 示例 2: 模拟较慢的推理（这里只是示例，不实际添加延迟）
            # detections = self._apply_latency_simulation(detections)

        return detections

    def get_model_info(self) -> Dict[str, Any]:
        """返回模型信息"""
        if self._inner_model is None:
            return self.model_info.copy()

        inner_info = self._inner_model.get_model_info()
        info = self.model_info.copy()
        info["params"] = inner_info.get("params", 0)
        info["simulated"] = True
        return info

    def warmup(self, image_size: tuple = (640, 640)) -> None:
        """模型预热"""
        if self._inner_model:
            self._inner_model.warmup(image_size)

    def set_simulation_mode(self, mode: str = "baseline") -> None:
        """
        设置模拟模式

        Args:
            mode: "baseline" 或 "custom"
                - baseline: 模拟基准模型（无变化）
                - custom: 模拟自定义模型（添加变化）
        """
        self.simulation_mode = mode

    @staticmethod
    def _apply_latency_simulation(
        detections: List[Detection], delay_ms: float = 50
    ) -> List[Detection]:
        """
        应用延迟模拟（仅用于演示）

        注意：这个函数只是概念性的，实际的延迟会在推理时自然产生
        """
        return detections

    def get_simulation_settings(self) -> Dict[str, Any]:
        """返回当前模拟设置"""
        return {
            "mode": self.simulation_mode,
            "description": "模拟模式：baseline（基准）或 custom（自定义）",
            "custom_adjustments": {
                "confidence_multiplier": 0.85,
                "latency_simulation": False,
            }
            if self.simulation_mode == "custom"
            else {},
        }


class UserModelLoader:
    """
    用户模型加载器

    支持多种用户模型格式：
    1. 配置文件（YAML）
    2. 权重文件（.pt, .pth, .onnx）
    3. 已注册的模型名（包括模拟模型）
    """

    @staticmethod
    def load_user_model(
        model_spec: str, device: str = "auto", conf_threshold: float = 0.001
    ) -> BaseModel:
        """
        加载用户模型

        Args:
            model_spec: 模型规格，可以是：
                - 模型名称（如 yolov8n:simulated）
                - 权重文件路径（如 /path/to/model.pt）
                - 配置文件路径（如 user_models/my_model.yaml）

        Returns:
            BaseModel 实例
        """
        from pathlib import Path
        import yaml

        # 检查是否是模拟模型
        if ":simulated" in model_spec.lower():
            model_name = model_spec.split(":")[0]
            simulated = SimulatedModel(device, conf_threshold)
            simulated.load_model(None, model_name)
            return simulated

        # 检查是否是配置文件
        config_path = Path(model_spec)
        if config_path.exists() and config_path.suffix in [".yaml", ".yml"]:
            return UserModelLoader._load_from_config(
                str(config_path), device, conf_threshold
            )

        # 检查是否是权重文件
        if config_path.exists() and config_path.suffix in [".pt", ".pth"]:
            return UserModelLoader._load_from_weights(
                str(config_path), device, conf_threshold
            )

        # 假设是已注册的模型名
        from . import create_model

        try:
            model = create_model(model_spec, device, conf_threshold)
            return model
        except ValueError:
            # 如果找不到，尝试使用模拟模型
            simulated = SimulatedModel(device, conf_threshold)
            simulated.load_model(None, f"{model_spec}:simulated")
            return simulated

    @staticmethod
    def _load_from_config(
        config_path: str, device: str, conf_threshold: float
    ) -> BaseModel:
        """从配置文件加载模型"""
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        model_name = config.get("name", "custom_model")
        model_type = config.get("model_type", "yolov8")

        # 根据类型创建模型
        from . import create_model

        # 创建基础模型
        base_model = create_model(model_type, device, conf_threshold)

        # 加载权重
        weights_path = config.get("weights")
        if weights_path:
            from . import load_model_wrapper

            load_model_wrapper(base_model, weights_path, model_name)

        # 覆盖模型信息
        base_model.model_info["name"] = model_name
        base_model.model_info["custom_config"] = config_path

        return base_model

    @staticmethod
    def _load_from_weights(
        weights_path: str, device: str, conf_threshold: float
    ) -> BaseModel:
        """从权重文件加载模型"""
        from . import create_model, load_model_wrapper

        # 默认使用 YOLOv8 框架
        model = create_model("yolov8n", device, conf_threshold)

        # 加载自定义权重
        load_model_wrapper(model, weights_path, "custom_model")

        # 更新模型信息
        model.model_info["name"] = f"custom_from_{Path(weights_path).stem}"
        model.model_info["custom_weights"] = weights_path

        return model

    @staticmethod
    def list_available_models(config_dir: str = "user_models") -> List[str]:
        """
        列出可用的用户模型

        Args:
            config_dir: 用户模型配置目录

        Returns:
            模型列表
        """
        from pathlib import Path
        import yaml

        models_dir = Path(config_dir)
        if not models_dir.exists():
            return []

        models = []

        for config_file in models_dir.glob("*.yaml"):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    name = config.get("name")
                    if name:
                        models.append(name)
            except Exception:
                continue

        return sorted(models)

    @staticmethod
    def create_model_config_template(
        name: str,
        description: str = "Custom object detection model",
        weights_path: str = "path/to/weights.pt",
        model_type: str = "yolov8",
        input_size: int = 640,
    ) -> str:
        """
        创建模型配置模板

        Args:
            name: 模型名称
            description: 描述
            weights_path: 权重文件路径
            model_type: 模型类型
            input_size: 输入尺寸

        Returns:
            YAML 配置字符串
        """
        config = f"""# User Model Configuration
name: {name}
description: {description}
framework: custom
model_type: {model_type}
weights: {weights_path}
input_size: {input_size}

# Optional: Specify class names
class_names:
  0: person
  1: car
  # ... add more classes

# Optional: Model-specific parameters
parameters:
  conf_threshold: 0.25
  iou_threshold: 0.5
"""
        return config
