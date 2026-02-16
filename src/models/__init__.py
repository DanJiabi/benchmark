from typing import Any, Optional
from .base import BaseModel
from .faster_rcnn import FasterRCNN
from .ultralytics_wrapper import UltralyticsWrapper
from .yolov11 import YOLOv11
from .simulated_model import SimulatedModel, UserModelLoader
from .onnx_model import ONNXModel
from .exporters import (
    ONNXExporter,
    TensorRTExporter,
    ExportManager,
    export_model_cli,
    batch_export_models,
)


ULTRALYTICS_MODELS = ["yolov8", "yolov9", "yolov10", "yolov11", "rtdetr", "rt-detr"]

SIMULATED_MODELS = ["simulated", "custom", "user"]


def create_model(
    model_name: str, device: str = "auto", conf_threshold: float = 0.001
) -> BaseModel:
    model_key = model_name.lower()

    # 检查 ONNX 模型
    if model_name.endswith(".onnx") or ":onnx" in model_key:
        onnx_model = ONNXModel(device, conf_threshold)
        # 如果包含路径分隔符，直接加载
        if "/" in model_name or "\\" in model_name or model_name.endswith(".onnx"):
            onnx_model.load_model(model_name.replace(":onnx", ""))
        return onnx_model

    # 检查 Ultralytics 模型
    for prefix in ULTRALYTICS_MODELS:
        if model_key.startswith(prefix):
            wrapper = UltralyticsWrapper(device, conf_threshold)
            wrapper.model_type = model_name
            return wrapper

    # 检查 FasterRCNN
    if model_key.startswith("faster"):
        return FasterRCNN(device, conf_threshold)

    # 检查模拟模型
    if model_key.startswith("simulated") or ":simulated" in model_key:
        simulated = SimulatedModel(device, conf_threshold)
        simulated.load_model(None, model_name)
        return simulated

    # 检查用户模型
    if any(key in model_key for key in SIMULATED_MODELS):
        return UserModelLoader.load_user_model(model_name, device, conf_threshold)

    raise ValueError(f"Unsupported model: {model_name}")


__all__ = [
    "create_model",
    "load_model_wrapper",
    "UserModelLoader",
    "BaseModel",
    "Detection",
    "ONNXModel",
    "ONNXExporter",
    "TensorRTExporter",
    "ExportManager",
    "export_model_cli",
    "batch_export_models",
]


def load_model_wrapper(
    model: BaseModel, weights_path: Optional[str], model_name: str
) -> None:
    if isinstance(model, UltralyticsWrapper):
        if weights_path is None:
            raise ValueError("Ultralytics models require weights_path")
        model.load_model(weights_path, model_name)
    elif isinstance(model, FasterRCNN):
        model.load_model(weights_path)
    elif isinstance(model, (ONNXModel, SimulatedModel)):
        if weights_path is not None:
            model.load_model(weights_path)
    else:
        if weights_path is not None:
            model.load_model(weights_path)
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
