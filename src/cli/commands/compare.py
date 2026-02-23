"""Compare command - Compare model formats.

This module implements the 'compare' command for od-benchmark CLI.
"""

from __future__ import annotations

from src.analysis import compare_model_formats_cli


def add_parser(subparsers):
    """Add compare command parser."""
    parser = subparsers.add_parser(
        "compare",
        help="Compare model performance across different formats (PyTorch vs ONNX)",
        description="Compare model performance across different formats",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model weights file (.pt)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name (default: auto-detect from filename)",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="pytorch,onnx",
        help="Formats to compare, comma-separated (default: pytorch,onnx)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path (default: config.yaml)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="Number of test images (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/format_comparison",
        help="Output directory (default: outputs/format_comparison)",
    )
    parser.set_defaults(command_func=main)
    return parser


def main(args):
    """Main compare function."""
    formats = [f.strip() for f in args.formats.split(",")]
    compare_model_formats_cli(
        model_path=args.model,
        model_name=args.model_name,
        formats=formats,
        config=args.config,
        num_images=args.num_images,
        output_dir=args.output_dir,
    )
