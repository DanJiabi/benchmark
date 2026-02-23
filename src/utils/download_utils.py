"""模型下载工具函数.

提供模型权重下载、缓存检查等功能的工具函数.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import requests
import yaml


def get_min_expected_size(file_name: str) -> int:
    """根据文件名获取最小预期大小（字节）.

    Args:
        file_name: 文件名

    Returns:
        最小预期大小（字节）
    """
    file_name = file_name.lower()

    # YOLO 模型文件大小参考
    size_map = {
        "yolov8n": 5 * 1024 * 1024,
        "yolov8s": 10 * 1024 * 1024,
        "yolov8m": 20 * 1024 * 1024,
        "yolov8l": 40 * 1024 * 1024,
        "yolov8x": 60 * 1024 * 1024,
        "yolov9t": 5 * 1024 * 1024,
        "yolov9s": 10 * 1024 * 1024,
        "yolov9m": 20 * 1024 * 1024,
        "yolov9c": 30 * 1024 * 1024,
        "yolov9e": 50 * 1024 * 1024,
        "yolov10n": 5 * 1024 * 1024,
        "yolov10s": 10 * 1024 * 1024,
        "yolov10m": 20 * 1024 * 1024,
        "yolov10b": 30 * 1024 * 1024,
        "yolov10l": 40 * 1024 * 1024,
        "yolov10x": 60 * 1024 * 1024,
        "yolo11n": 5 * 1024 * 1024,
        "yolo11s": 10 * 1024 * 1024,
        "yolo11m": 20 * 1024 * 1024,
        "yolo11l": 40 * 1024 * 1024,
        "yolo11x": 60 * 1024 * 1024,
        "yolo26n": 5 * 1024 * 1024,
        "yolo26s": 10 * 1024 * 1024,
        "yolo26m": 20 * 1024 * 1024,
        "yolo26l": 40 * 1024 * 1024,
        "yolo26x": 60 * 1024 * 1024,
        "rtdetr-l": 50 * 1024 * 1024,
        "rtdetr-x": 100 * 1024 * 1024,
    }

    for key, size in size_map.items():
        if key in file_name:
            return size

    # 默认最小大小：至少 100KB
    return 100 * 1024


def check_file_complete(file_path: Path, expected_size: int = None) -> Tuple[bool, str]:
    """检查文件是否完整.

    Args:
        file_path: 文件路径
        expected_size: 预期文件大小

    Returns:
        (是否完整, 状态信息)
    """
    if not file_path.exists():
        return False, "文件不存在"

    file_size = file_path.stat().st_size

    # 检查文件是否为空
    if file_size == 0:
        return False, "文件为空"

    # 检查文件是否过小（不完整）
    min_size = get_min_expected_size(file_path.name)
    if file_size < min_size:
        size_mb = file_size / 1024 / 1024
        min_mb = min_size / 1024 / 1024
        return False, f"文件过小 ({size_mb:.2f} MB < {min_mb:.2f} MB)"

    # 检查文件大小是否匹配预期
    if expected_size and file_size < expected_size * 0.95:
        return False, f"文件大小不匹配"

    # 检查文件是否可以读取
    try:
        with open(file_path, "rb") as f:
            data = f.read(1024)
            if len(data) < 10:
                return False, "文件内容异常"
    except Exception as e:
        return False, f"文件读取失败"

    # 尝试加载验证（可选，需要安装 torch）
    try:
        import torch

        try:
            checkpoint = torch.load(file_path, map_location="cpu", weights_only=True)
            if isinstance(checkpoint, dict):
                valid_keys = ["model", "state_dict", "ema", "model_state_dict"]
                has_valid_key = any(key in checkpoint for key in valid_keys)
                if has_valid_key or len(checkpoint) > 0:
                    return True, "文件完整"
            elif hasattr(checkpoint, "model"):
                return True, "文件完整"
        except Exception:
            try:
                checkpoint = torch.load(
                    file_path, map_location="cpu", weights_only=False
                )
                return True, "文件完整"
            except Exception:
                return False, "文件加载失败"
    except ImportError:
        # 未安装 torch，跳过加载验证
        pass

    return True, "文件完整"


def download_file(
    url: str,
    output_path: Path,
    expected_size: int = None,
    overwrite: bool = False,
    logger=None,
) -> bool:
    """下载文件.

    Args:
        url: 下载URL
        output_path: 保存路径
        expected_size: 预期文件大小
        overwrite: 是否覆盖已存在文件
        logger: 日志记录器（可选）

    Returns:
        下载是否成功
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 检查文件是否已存在且完整
    if not overwrite and output_path.exists():
        is_complete, message = check_file_complete(output_path, expected_size)
        if is_complete:
            if logger:
                logger.info(f"  ✅ 文件已存在且完整: {output_path.name}")
            else:
                print(f"  ✅ 文件已存在且完整: {output_path.name}")
            return True
        else:
            if logger:
                logger.warning(f"  ⚠️  文件不完整 ({message}), 将重新下载")
            else:
                print(f"  ⚠️  文件不完整 ({message}), 将重新下载")

    if logger:
        logger.info(f"  📥 下载: {url}")
        logger.info(f"  📁 保存到: {output_path}")
    else:
        print(f"  📥 下载: {url}")
        print(f"  📁 保存到: {output_path}")

    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded_size = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)

                    # 显示进度
                    if total_size > 0 and not logger:
                        progress = (downloaded_size / total_size) * 100
                        print(
                            f"\r  ⏳ 进度: {progress:.1f}% ({downloaded_size}/{total_size})",
                            end="",
                        )

        if not logger:
            print()  # 换行

        # 验证下载的文件
        if total_size > 0 and downloaded_size != total_size:
            msg = f"下载大小不匹配: {downloaded_size}/{total_size}"
            if logger:
                logger.error(f"  ⚠️  {msg}")
            else:
                print(f"  ⚠️  {msg}")
            output_path.unlink()
            return False

        msg = f"下载完成: {output_path.name} ({downloaded_size / 1024 / 1024:.2f} MB)"
        if logger:
            logger.info(f"  ✅ {msg}")
        else:
            print(f"  ✅ {msg}")
        return True

    except Exception as e:
        msg = f"下载失败: {e}"
        if logger:
            logger.error(f"  ❌ {msg}")
        else:
            print(f"  ❌ {msg}")
        if output_path.exists():
            output_path.unlink()
        return False


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件.

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """保存配置文件.

    Args:
        config: 配置字典
        config_path: 配置文件路径
    """
    config_file = Path(config_path)

    # 创建备份
    if config_file.exists():
        backup_path = config_file.with_suffix(".yaml.bak")
        backup_path.write_text(config_file.read_text(), encoding="utf-8")

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(
            config, f, default_flow_style=False, allow_unicode=True, sort_keys=False
        )


def get_available_models(config_path: str = "config.yaml") -> List[Dict[str, Any]]:
    """获取配置文件中所有可用模型.

    Args:
        config_path: 配置文件路径

    Returns:
        模型配置列表
    """
    config = load_config(config_path)
    return config.get("models", [])


def check_model_cached(
    model_name: str,
    config_path: str = "config.yaml",
    cache_dir: str = "models_cache",
) -> Tuple[bool, Optional[str]]:
    """检查模型是否已缓存.

    Args:
        model_name: 模型名称
        config_path: 配置文件路径
        cache_dir: 缓存目录

    Returns:
        (是否已缓存, 状态信息)
    """
    models = get_available_models(config_path)

    for model in models:
        if model.get("name") == model_name:
            weights = model.get("weights")
            if weights is None:
                return True, "内置模型"

            weights_path = Path(cache_dir) / weights
            if weights_path.exists():
                is_complete, message = check_file_complete(weights_path)
                if is_complete:
                    size_mb = weights_path.stat().st_size / 1024 / 1024
                    return True, f"已缓存 ({size_mb:.2f} MB)"
                else:
                    return False, f"缓存不完整: {message}"
            else:
                return False, "未下载"

    return False, "模型不存在"


def download_model(
    model_name: str,
    config_path: str = "config.yaml",
    cache_dir: str = "models_cache",
    overwrite: bool = False,
    logger=None,
) -> bool:
    """下载指定模型.

    Args:
        model_name: 模型名称
        config_path: 配置文件路径
        cache_dir: 缓存目录
        overwrite: 是否覆盖已存在文件
        logger: 日志记录器（可选）

    Returns:
        下载是否成功
    """
    models = get_available_models(config_path)
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    for model in models:
        if model.get("name") == model_name:
            weights = model.get("weights")
            url = model.get("url")

            if weights is None:
                if logger:
                    logger.info(
                        f"  ℹ️  {model_name}: 无权重文件（可能使用内置预训练权重）"
                    )
                else:
                    print(f"  ℹ️  {model_name}: 无权重文件（可能使用内置预训练权重）")
                return True

            if not url:
                if logger:
                    logger.error(f"  ⚠️  {model_name}: 未提供下载 URL")
                else:
                    print(f"  ⚠️  {model_name}: 未提供下载 URL")
                return False

            weights_path = cache_path / weights
            return download_file(url, weights_path, overwrite=overwrite, logger=logger)

    if logger:
        logger.error(f"  ❌ 未找到模型: {model_name}")
    else:
        print(f"  ❌ 未找到模型: {model_name}")
    return False


def get_model_info(
    model_name: str,
    config_path: str = "config.yaml",
    cache_dir: str = "models_cache",
) -> Optional[Dict[str, Any]]:
    """获取模型详细信息.

    Args:
        model_name: 模型名称
        config_path: 配置文件路径
        cache_dir: 缓存目录

    Returns:
        模型信息字典，如果模型不存在则返回 None
    """
    models = get_available_models(config_path)

    for model in models:
        if model.get("name") == model_name:
            cached, cache_status = check_model_cached(
                model_name, config_path, cache_dir
            )

            info = model.copy()
            info["cached"] = cached
            info["cache_status"] = cache_status

            # 添加文件大小信息
            weights = model.get("weights")
            if weights:
                weights_path = Path(cache_dir) / weights
                if weights_path.exists():
                    info["file_size_mb"] = weights_path.stat().st_size / 1024 / 1024

            return info

    return None


def add_model_to_config(
    name: str,
    framework: str,
    weights: Optional[str] = None,
    url: Optional[str] = None,
    config_path: str = "config.yaml",
) -> bool:
    """添加模型到配置文件.

    Args:
        name: 模型名称
        framework: 框架类型
        weights: 权重文件名
        url: 下载URL
        config_path: 配置文件路径

    Returns:
        添加是否成功

    Raises:
        ValueError: 模型名称已存在
    """
    config = load_config(config_path)

    if "models" not in config:
        config["models"] = []

    # 检查名称是否已存在
    for model in config["models"]:
        if model.get("name") == name:
            raise ValueError(f"模型名称已存在: {name}")

    # 创建新模型配置
    new_model = {
        "name": name,
        "framework": framework,
    }

    if weights:
        new_model["weights"] = weights

    if url:
        new_model["url"] = url

    # 添加到列表
    config["models"].append(new_model)

    # 保存配置
    save_config(config, config_path)

    return True


def remove_model_from_config(
    name: str,
    config_path: str = "config.yaml",
    delete_weights: bool = False,
    cache_dir: str = "models_cache",
) -> bool:
    """从配置文件删除模型.

    Args:
        name: 模型名称
        config_path: 配置文件路径
        delete_weights: 是否同时删除本地权重文件
        cache_dir: 缓存目录

    Returns:
        删除是否成功

    Raises:
        ValueError: 模型不存在
    """
    config = load_config(config_path)

    if "models" not in config:
        raise ValueError("配置文件中没有模型")

    # 查找并删除模型
    model_to_remove = None
    for model in config["models"]:
        if model.get("name") == name:
            model_to_remove = model
            break

    if model_to_remove is None:
        raise ValueError(f"模型不存在: {name}")

    # 从列表中移除
    config["models"].remove(model_to_remove)

    # 保存配置
    save_config(config, config_path)

    # 可选：删除权重文件
    if delete_weights:
        weights = model_to_remove.get("weights")
        if weights:
            weights_path = Path(cache_dir) / weights
            if weights_path.exists():
                weights_path.unlink()

    return True


def detect_framework(model_name: str) -> str:
    """根据模型名称自动检测框架类型.

    Args:
        model_name: 模型名称

    Returns:
        框架类型
    """
    name_lower = model_name.lower()

    if "faster_rcnn" in name_lower:
        return "torchvision"
    elif name_lower.endswith(".onnx"):
        return "onnx"
    else:
        # 默认为 ultralytics（YOLO系列）
        return "ultralytics"
