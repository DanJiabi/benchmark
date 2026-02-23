"""Model management command group.

This module implements the 'model' command group for od-benchmark CLI.
"""

from __future__ import annotations

from src.utils import (
    add_model_to_config,
    remove_model_from_config,
    get_model_info,
    detect_framework,
)


def add_parser(subparsers):
    """Add model command group parser."""
    parser = subparsers.add_parser(
        "model",
        help="Manage models in configuration",
        description="Add, remove, or view model configurations",
    )
    model_subparsers = parser.add_subparsers(
        dest="model_command",
        help="Model management commands",
        required=True,
    )

    # Add command
    add_cmd = model_subparsers.add_parser(
        "add",
        help="Add a new model to configuration",
    )
    add_cmd.add_argument(
        "--name",
        type=str,
        required=True,
        help="Model name (unique identifier)",
    )
    add_cmd.add_argument(
        "--framework",
        type=str,
        default=None,
        help="Framework type (ultralytics, torchvision, onnx, custom). Auto-detected if not specified.",
    )
    add_cmd.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Weights filename",
    )
    add_cmd.add_argument(
        "--url",
        type=str,
        default=None,
        help="Download URL for weights",
    )
    add_cmd.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path (default: config.yaml)",
    )

    # Remove command
    remove_cmd = model_subparsers.add_parser(
        "remove",
        help="Remove a model from configuration",
    )
    remove_cmd.add_argument(
        "--name",
        type=str,
        required=True,
        help="Model name to remove",
    )
    remove_cmd.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )
    remove_cmd.add_argument(
        "--delete-weights",
        action="store_true",
        help="Also delete local weights file",
    )
    remove_cmd.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path (default: config.yaml)",
    )
    remove_cmd.add_argument(
        "--cache-dir",
        type=str,
        default="models_cache",
        help="Cache directory (default: models_cache)",
    )

    # Info command
    info_cmd = model_subparsers.add_parser(
        "info",
        help="Show detailed information about a model",
    )
    info_cmd.add_argument(
        "--name",
        type=str,
        required=True,
        help="Model name",
    )
    info_cmd.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path (default: config.yaml)",
    )
    info_cmd.add_argument(
        "--cache-dir",
        type=str,
        default="models_cache",
        help="Cache directory (default: models_cache)",
    )

    parser.set_defaults(command_func=main)
    return parser


def main(args):
    """Model management main function."""
    if args.model_command == "add":
        _handle_add(args)
    elif args.model_command == "remove":
        _handle_remove(args)
    elif args.model_command == "info":
        _handle_info(args)


def _handle_add(args):
    """Handle model add command."""
    # Auto-detect framework if not specified
    framework = args.framework
    if framework is None:
        framework = detect_framework(args.name)
        print(f"📝 自动检测到框架: {framework}")

    try:
        add_model_to_config(
            name=args.name,
            framework=framework,
            weights=args.weights,
            url=args.url,
            config_path=args.config,
        )
        print(f"✅ 成功添加模型: {args.name}")
        print(f"   框架: {framework}")
        if args.weights:
            print(f"   权重: {args.weights}")
        if args.url:
            print(f"   URL: {args.url}")
        print(f"   配置文件已备份: {args.config}.bak")
    except ValueError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ 添加失败: {e}")


def _handle_remove(args):
    """Handle model remove command."""
    # Confirm deletion
    if not args.force:
        confirm = input(f"⚠️  确认删除模型 '{args.name}'? [y/N]: ")
        if confirm.lower() != "y":
            print("已取消删除")
            return

    try:
        remove_model_from_config(
            name=args.name,
            config_path=args.config,
            delete_weights=args.delete_weights,
            cache_dir=args.cache_dir,
        )
        print(f"✅ 成功删除模型: {args.name}")
        if args.delete_weights:
            print("   权重文件已删除")
        print(f"   配置文件已备份: {args.config}.bak")
    except ValueError as e:
        print(f"❌ {e}")
    except Exception as e:
        print(f"❌ 删除失败: {e}")


def _handle_info(args):
    """Handle model info command."""
    info = get_model_info(args.name, args.config, args.cache_dir)

    if info is None:
        print(f"❌ 未找到模型: {args.name}")
        return

    print("=" * 80)
    print("模型详细信息")
    print("=" * 80)
    print(f"名称:     {info.get('name')}")
    print(f"框架:     {info.get('framework')}")

    weights = info.get("weights")
    if weights:
        print(f"权重:     {weights}")
    else:
        print(f"权重:     内置模型")

    url = info.get("url")
    if url:
        print(f"URL:      {url}")

    print(f"状态:     {info.get('cache_status')}")

    if info.get("file_size_mb"):
        print(f"文件大小: {info['file_size_mb']:.2f} MB")

    print("=" * 80)
