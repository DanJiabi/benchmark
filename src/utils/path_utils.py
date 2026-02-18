"""路径处理工具函数"""

from pathlib import Path
from typing import Union


def resolve_model_path(weights_file: str, cache_dir: str = "models_cache") -> Path:
    """解析模型权重路径

    根据输入的权重文件路径，自动判断是完整路径还是文件名，
    并返回正确的 Path 对象。

    Args:
        weights_file: 权重文件路径或文件名
            - 如果包含目录分隔符（/ 或 \\），视为完整路径
            - 否则视为文件名，会在 cache_dir 中查找
        cache_dir: 缓存目录名称，默认为 "models_cache"

    Returns:
        解析后的完整路径 Path 对象

    Examples:
        >>> resolve_model_path("yolov8n.pt")
        Path("models_cache/yolov8n.pt")

        >>> resolve_model_path("models_export/yolov8n.onnx")
        Path("models_export/yolov8n.onnx")

        >>> resolve_model_path("/absolute/path/model.pt")
        Path("/absolute/path/model.pt")

        >>> resolve_model_path("model.pt", cache_dir="custom_cache")
        Path("custom_cache/model.pt")
    """
    if "/" in weights_file or "\\" in weights_file:
        return Path(weights_file)

    return Path(cache_dir) / weights_file


def ensure_parent_dir(path: Union[str, Path]) -> Path:
    """确保父目录存在

    创建路径的父目录（如果不存在），返回 Path 对象。

    Args:
        path: 文件路径（字符串或 Path 对象）

    Returns:
        Path 对象

    Examples:
        >>> path = ensure_parent_dir("outputs/results/model.json")
        >>> path.parent.exists()
        True

        >>> path = ensure_parent_dir(Path("data/cache/temp.txt"))
        >>> path.parent.exists()
        True
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
