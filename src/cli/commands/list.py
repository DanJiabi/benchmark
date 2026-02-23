"""List command - Display available models.

This module implements the 'list' command for od-benchmark CLI.
"""

from __future__ import annotations

from src.utils import get_available_models, check_model_cached


def add_parser(subparsers):
    """Add list command parser."""
    parser = subparsers.add_parser(
        "list",
        help="List all available models in config",
        description="Display all models configured in config.yaml with their cache status",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path (default: config.yaml)",
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Show only cached models",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default=None,
        help="Filter by framework (ultralytics, torchvision, onnx)",
    )
    parser.set_defaults(command_func=main)
    return parser


def main(args):
    """List all available models."""
    try:
        models = get_available_models(args.config)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # Filter by framework if specified
    if args.framework:
        models = [m for m in models if m.get("framework") == args.framework]

    # Filter by cached status if specified
    if args.cached:
        cached_models = []
        for model in models:
            model_name = model.get("name")
            if model_name:
                cached, _ = check_model_cached(str(model_name), args.config)
                if cached:
                    cached_models.append(model)
        models = cached_models

    if not models:
        print("未找到符合条件的模型")
        return

    print("=" * 80)
    print("可用模型列表")
    print("=" * 80)
    print(f"总计: {len(models)} 个模型")
    print()

    # Group by series
    series_groups = {}

    for model in models:
        name = model.get("name", "unknown")
        framework = model.get("framework", "unknown")

        # Detect series
        if "yolo26" in name.lower():
            series = "YOLO26 系列"
        elif "yolo11" in name.lower():
            series = "YOLO11 系列"
        elif "yolov10" in name.lower():
            series = "YOLOv10 系列"
        elif "yolov9" in name.lower():
            series = "YOLOv9 系列"
        elif "yolov8" in name.lower():
            series = "YOLOv8 系列"
        elif "rtdetr" in name.lower():
            series = "RT-DETR 系列"
        elif "faster_rcnn" in name.lower():
            series = "Faster R-CNN"
        else:
            series = "其他"

        if series not in series_groups:
            series_groups[series] = []
        series_groups[series].append((name, framework, model))

    # Display by series
    series_order = [
        "YOLO26 系列",
        "YOLO11 系列",
        "YOLOv10 系列",
        "YOLOv9 系列",
        "YOLOv8 系列",
        "RT-DETR 系列",
        "Faster R-CNN",
        "其他",
    ]

    for series_name in series_order:
        if series_name not in series_groups:
            continue

        models_in_series = series_groups[series_name]
        print(f"\n{series_name}:")
        print("-" * 80)

        for name, framework, model in sorted(models_in_series, key=lambda x: x[0]):
            cached, cache_status = check_model_cached(name, args.config)
            status_icon = "✓" if cached else "✗"

            weights = model.get("weights")
            if weights is None:
                cache_status = "内置模型"

            print(f"  {status_icon} {name:<15} ({cache_status:<20}) {framework}")

    print()
    print("=" * 80)
    print("图例: ✓ = 已缓存/可用  ✗ = 未下载")
    print("=" * 80)
