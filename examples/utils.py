"""示例脚本通用工具函数

提供模型加载、输出设置等通用功能，简化示例脚本的编写
"""

from typing import Any, Tuple, Optional
from pathlib import Path

from src.models import create_model, load_model_wrapper
from src.utils import resolve_model_path


def load_model_for_demo(
    model_name: str,
    device: str = "auto",
    conf_threshold: float = 0.25,
    verbose: bool = True,
) -> Tuple[Any, str]:
    """示例专用的模型加载函数

    封装了路径解析、错误处理、状态打印，适用于示例脚本。

    Args:
        model_name: 模型名称或路径
            - 可以是模型名称（如 "yolov8n"）
            - 可以是权重文件名（如 "yolov8n.pt"）
            - 可以是完整路径（如 "models_export/yolov8n.onnx"）
        device: 设备类型，默认 "auto"（自动选择）
        conf_threshold: 置信度阈值，默认 0.25
        verbose: 是否打印详细信息，默认 True

    Returns:
        (model, model_name) 元组

    Raises:
        FileNotFoundError: 模型文件不存在
        ValueError: 不支持的模型类型

    Examples:
        >>> # 加载 PyTorch 模型
        >>> model, name = load_model_for_demo("yolov8n")
        >>> print(f"加载模型: {name}")

        >>> # 加载 ONNX 模型
        >>> model, name = load_model_for_demo("models_export/yolov8n.onnx")

        >>> # 静默加载（不打印信息）
        >>> model, name = load_model_for_demo("yolov8n", verbose=False)
    """
    if verbose:
        print(f"加载模型: {model_name}")

    # 解析权重路径
    model_key = model_name.lower()

    # 判断是否是 ONNX 模型
    is_onnx = model_name.endswith(".onnx") or ":onnx" in model_key

    if is_onnx:
        # ONNX 模型：直接使用路径
        weights_path = Path(model_name.replace(":onnx", ""))
    else:
        # PyTorch 模型：使用工具函数解析路径
        # 移除可能的 .pt 后缀（如果有）
        clean_name = model_name.replace(".pt", "")
        weights_path = resolve_model_path(f"{clean_name}.pt")

    # 检查文件是否存在
    if not weights_path.exists():
        raise FileNotFoundError(
            f"模型文件不存在: {weights_path}\n请检查路径或先下载模型权重"
        )

    # 创建模型
    if is_onnx:
        # ONNX 模型：使用完整路径创建
        model = create_model(
            str(weights_path), device=device, conf_threshold=conf_threshold
        )
        model_display_name = weights_path.stem
    else:
        # PyTorch 模型：使用模型名称创建
        model = create_model(
            weights_path.stem, device=device, conf_threshold=conf_threshold
        )
        # 加载权重
        load_model_wrapper(model, str(weights_path), weights_path.stem)
        model_display_name = weights_path.stem

    if verbose:
        print(f"  ✅ 加载成功")
        info = model.get_model_info()
        print(f"     类型: {info.get('name', 'N/A')}")
        print(f"     设备: {info.get('device', 'N/A')}")
        if "params" in info:
            print(f"     参数: {info['params']:.2f}M")

    return model, model_display_name


def setup_demo_output(demo_name: str, base_dir: str = "outputs/examples") -> Path:
    """设置示例输出目录

    创建标准的示例输出目录结构，确保目录存在。

    Args:
        demo_name: 示例名称（用于创建子目录）
        base_dir: 基础输出目录，默认 "outputs/examples"

    Returns:
        输出目录路径

    Examples:
        >>> output_dir = setup_demo_output("compare_pt_onnx")
        >>> print(f"输出目录: {output_dir}")
        输出目录: outputs/examples/compare_pt_onnx

        >>> # 保存结果
        >>> result_file = output_dir / "result.json"
    """
    from src.utils import ensure_parent_dir

    output_dir = Path(base_dir) / demo_name
    # 创建目录（使用 ensure_parent_dir 确保）
    ensure_parent_dir(output_dir / "dummy.txt")

    print(f"输出目录: {output_dir}")
    return output_dir


def print_demo_header(title: str, width: int = 80):
    """打印示例标题

    Args:
        title: 标题文本
        width: 宽度，默认 80
    """
    print("=" * width)
    print(title)
    print("=" * width)


def print_demo_footer(title: str = "完成", width: int = 80):
    """打印示例结束信息

    Args:
        title: 结束文本，默认 "完成"
        width: 宽度，默认 80
    """
    print("=" * width)
    print(title)
    print("=" * width)
