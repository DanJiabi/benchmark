"""Download command - Download model weights.

This module implements the 'download' command for od-benchmark CLI.
"""

from __future__ import annotations

from src.utils import get_available_models, download_model


def add_parser(subparsers):
    """Add download command parser."""
    parser = subparsers.add_parser(
        "download",
        help="Download model weights",
        description="Download model weights from configured URLs",
    )
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        help="Model name(s) to download (can be used multiple times)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="download_all",
        help="Download all models",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path (default: config.yaml)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="models_cache",
        help="Cache directory (default: models_cache)",
    )
    parser.set_defaults(command_func=main)
    return parser


def main(args):
    """Download model weights."""
    if not args.model and not args.download_all:
        print("❌ 必须指定 --model 或 --all")
        print("示例:")
        print("  od-benchmark download --model yolov8n")
        print("  od-benchmark download --model yolov8n --model yolov8s")
        print("  od-benchmark download --all")
        return

    try:
        models = get_available_models(args.config)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # Determine which models to download
    if args.download_all:
        models_to_download = [m.get("name") for m in models if m.get("weights")]
    else:
        models_to_download = args.model

    if not models_to_download:
        print("❌ 没有可下载的模型")
        return

    print("=" * 80)
    print("模型权重下载")
    print("=" * 80)
    print(f"配置文件: {args.config}")
    print(f"缓存目录: {args.cache_dir}")
    print(f"模型数量: {len(models_to_download)}")
    print("=" * 80)

    success_count = 0
    fail_count = 0

    for idx, model_name in enumerate(models_to_download, 1):
        if not model_name:
            continue

        print(f"\n[{idx}/{len(models_to_download)}] 处理: {model_name}")
        print("-" * 80)

        # Check if model exists in config
        model_exists = any(m.get("name") == model_name for m in models)
        if not model_exists:
            print(f"  ❌ 配置文件中未找到模型: {model_name}")
            fail_count += 1
            continue

        # Download
        success = download_model(
            str(model_name),
            config_path=args.config,
            cache_dir=args.cache_dir,
            overwrite=args.force,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1

    print()
    print("=" * 80)
    print("下载完成")
    print(f"  ✅ 成功: {success_count}")
    print(f"  ❌ 失败: {fail_count}")
    print("=" * 80)
