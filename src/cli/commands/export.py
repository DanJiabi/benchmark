"""Export command - Export models to ONNX/TensorRT.

This module implements the 'export' command for od-benchmark CLI.
"""

from __future__ import annotations

from src.models import batch_export_models


def add_parser(subparsers):
    """Add export command parser."""
    parser = subparsers.add_parser(
        "export",
        help="Export models to ONNX or TensorRT format",
        description="Export models (YOLO, RT-DETR, Faster R-CNN) to ONNX or TensorRT format for optimized inference",
    )
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        help="Path to model weights file(s) (.pt), or 'faster_rcnn' for torchvision pretrained. Can be used multiple times",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Export all models from models_cache directory",
    )
    parser.add_argument(
        "--include-faster-rcnn",
        action="store_true",
        help="Include Faster R-CNN (torchvision pretrained) when using --all-models",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "tensorrt", "all"],
        help="Export format (default: onnx)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models_export",
        help="Output directory (default: models_export)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("H", "W"),
        help="Input image size (default: 640 640)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Use dynamic input size (ONNX only)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Simplify ONNX model (default: True)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 precision (TensorRT only, default: True)",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Use INT8 quantization (TensorRT only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for export (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for export (default: cpu)",
    )
    parser.set_defaults(command_func=main)
    return parser


def main(args):
    """Main export function."""
    # Validate arguments
    if not args.model and not args.all_models:
        print("错误: 必须指定 --model 或 --all-models")
        print("示例:")
        print("  od-benchmark export --model model.pt")
        print("  od-benchmark export --model model1.pt --model model2.pt")
        print("  od-benchmark export --all-models")
        print("  od-benchmark export --model faster_rcnn  # 导出 Faster R-CNN")
        print("  od-benchmark export --all-models --include-faster-rcnn")
        return

    batch_export_models(
        model_paths=args.model or [],
        all_models=args.all_models,
        format=args.format,
        output_dir=args.output_dir,
        input_size=tuple(args.input_size),
        dynamic=args.dynamic,
        simplify=args.simplify,
        fp16=args.fp16,
        int8=args.int8,
        batch_size=args.batch_size,
        device=args.device,
        include_faster_rcnn=getattr(args, "include_faster_rcnn", False),
    )
