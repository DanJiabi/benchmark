"""Analyze command - Compare models.

This module implements the 'analyze' command for od-benchmark CLI.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.models import create_model, load_model_wrapper, UserModelLoader
from src.data.coco_dataset import COCOInferenceDataset
from src.utils.logger import Config, setup_logger
from src.analysis import ModelComparison


def add_parser(subparsers):
    """Add analyze command parser."""
    parser = subparsers.add_parser(
        "analyze",
        help="Compare baseline model with user model",
        description="Compare baseline model performance with user custom model",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        action="append",
        help="Baseline model name(s) from config.yaml (can be used multiple times)",
    )
    parser.add_argument(
        "--all-baselines",
        action="store_true",
        help="Use all configured baseline models",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        action="append",
        help="User model(s)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="Number of test images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/analysis",
        help="Output directory",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="all",
        choices=["json", "html", "csv", "all"],
        help="Output format",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )
    parser.set_defaults(command_func=main)
    return parser


def main(args):
    """Main analyze function."""
    try:
        config = Config(args.config)
        logger = setup_logger(config)
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {args.config}")
        print("   请检查配置文件路径")
        return
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        print(f"   文件: {args.config}")
        return

    logger.info("=" * 70)
    logger.info("模型对比分析")
    logger.info("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config to get models
    models_config = config.get_models_config()
    dataset_config = config.get_dataset_config()
    dataset_path = dataset_config["path"]
    split = dataset_config["split"]

    logger.info(f"加载数据集: {dataset_path}/{split}")
    dataset = COCOInferenceDataset(dataset_path, split)
    logger.info(f"数据集大小: {len(dataset)} 张图片")

    # Get baseline models
    baseline_models = []
    if args.all_baselines:
        baseline_models = models_config
        logger.info(f"使用所有基准模型: {[m['name'] for m in baseline_models]}")
    elif args.baseline:
        baseline_names = args.baseline
        for baseline_name in baseline_names:
            for model_cfg in models_config:
                if model_cfg["name"] == baseline_name:
                    baseline_models.append(model_cfg)
                    break
        logger.info(f"使用基准模型: {baseline_names}")
    else:
        logger.error("❌ 必须指定 --baseline 或 --all-baselines")
        logger.info(f"可用的模型: {', '.join([m['name'] for m in models_config])}")
        return

    if not baseline_models:
        logger.error("❌ 未找到基准模型")
        logger.info(f"可用的模型: {', '.join([m['name'] for m in models_config])}")
        return

    # Get user models
    user_models = args.user_model if args.user_model else []
    logger.info(f"用户模型: {user_models}")

    conf_threshold = 0.001

    # Get annotations file
    annotations_file = (
        Path(dataset_path).expanduser() / "annotations" / f"instances_{split}.json"
    )

    # Run all comparisons
    all_comparisons = []
    for baseline_config in baseline_models:
        baseline_name = baseline_config["name"]
        logger.info(f"加载基准模型: {baseline_name}")

        try:
            baseline_model = create_model(
                baseline_name, device="auto", conf_threshold=conf_threshold
            )
            weights_file = baseline_config.get("weights")
            if weights_file:
                load_model_wrapper(
                    baseline_model,
                    str(Path("models_cache") / weights_file),
                    baseline_name,
                )
        except ValueError as e:
            logger.error(f"❌ 基准模型 {baseline_name} 加载失败: {e}")
            continue

        # Compare with each user model
        for user_model_spec in user_models:
            logger.info(f"  对比用户模型: {user_model_spec}")

            try:
                user_model = UserModelLoader.load_user_model(
                    user_model_spec, device="auto", conf_threshold=conf_threshold
                )

                comparison = ModelComparison(baseline_model, user_model, logger)
                comparison.run_comparison(
                    dataset=dataset,
                    annotations_file=str(annotations_file),
                    max_images=args.num_images,
                    conf_threshold=conf_threshold,
                )

                comparison_result = comparison.get_comparison()
                comparison_result["baseline_name"] = baseline_name
                comparison_result["user_model_spec"] = user_model_spec
                comparison_result["timestamp"] = datetime.now().isoformat()

                all_comparisons.append(comparison_result)

            except Exception as e:
                logger.error(f"  ❌ 用户模型 {user_model_spec} 对比失败: {e}")
                import traceback

                traceback.print_exc()
                continue

    if not all_comparisons:
        logger.error("❌ 没有成功的对比分析")
        return

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_comparisons": len(all_comparisons),
        "baseline_models": [m["name"] for m in baseline_models],
        "user_models": user_models,
        "comparisons": all_comparisons,
    }

    # Determine formats
    formats = []
    if args.format == "all":
        formats = ["json", "html", "csv"]
    else:
        formats = [args.format]

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"汇总结果已保存: {summary_file}")

    # Save individual results
    for idx, comp in enumerate(all_comparisons):
        comp_dir = output_dir / f"comparison_{idx:03d}"
        comp_dir.mkdir(parents=True, exist_ok=True)

        comp_file = comp_dir / "comparison.json"
        with open(comp_file, "w", encoding="utf-8") as f:
            json.dump(comp, f, indent=2, ensure_ascii=False)

    logger.info("")
    logger.info("=" * 70)
    logger.info("所有对比分析完成！")
    logger.info(f"汇总: {summary_file}")
    logger.info(f"对比结果: {output_dir}")
    logger.info("=" * 70)
