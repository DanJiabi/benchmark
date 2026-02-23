"""Benchmark command - Run model evaluation.

This module implements the 'benchmark' command for od-benchmark CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

from src.models.base import Detection
from src.models import (
    create_model,
    load_model_wrapper,
)
from src.data.coco_dataset import COCOInferenceDataset
from src.metrics.coco_metrics import COCOMetrics, PerformanceMetrics, MetricsAggregator
from src.utils.logger import Config, setup_logger
from src.utils.visualization import (
    save_detection_visualization,
    plot_metrics_comparison,
    plot_fps_vs_map,
    plot_model_size_vs_performance,
    generate_results_table,
)
from src.utils import resolve_model_path
from src.utils.logger import download_model_weights


def add_parser(subparsers):
    """Add benchmark command parser."""
    parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmark evaluation",
        description="Run object detection model benchmark evaluation",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path (default: config.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        help="Specify model(s) to test (can be used multiple times)",
    )
    parser.add_argument("--all", action="store_true", help="Test all configured models")
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of test images (default: all data)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable detection box visualization",
    )
    parser.add_argument(
        "--num-viz-images",
        type=int,
        default=10,
        help="Number of visualization images (default: 10)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Confidence threshold (default: use config file value)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/results",
        help="Output directory (default: outputs/results)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pytorch",
        choices=["pytorch", "onnx"],
        help="Model format (default: pytorch)",
    )
    parser.set_defaults(command_func=main)
    return parser


def run_single_model(
    model_config: Dict[str, Any],
    dataset: COCOInferenceDataset,
    coco_metrics_calculator: COCOMetrics,
    logger,
    max_images: Optional[int] = None,
    conf_threshold: float = 0.001,
    visualize: bool = False,
    vis_dir: Optional[Path] = None,
    num_viz_images: int = 10,
) -> Optional[Dict[str, Any]]:
    """Run single model evaluation."""
    import numpy as np
    from tqdm import tqdm

    model_name = model_config["name"]
    framework = model_config["framework"]

    # Handle ONNX model path
    if framework == "onnx":
        weights_path = Path(model_config["path"])
        weights_file = str(weights_path)
        weights_url = None
    else:
        weights_file = model_config["weights"]
        weights_url = model_config.get("url")
        weights_path = None

    logger.info(f"开始评估模型: {model_name}")

    # Process weights path (PyTorch only)
    if framework != "onnx" and weights_file:
        weights_path = resolve_model_path(weights_file)

        if not weights_path.exists() and weights_url:
            logger.info(f"下载模型权重: {weights_url}")
            try:
                download_model_weights(weights_url, weights_path)
            except Exception as e:
                logger.error(f"❌ 模型下载失败: {model_name}")
                logger.error(f"   URL: {weights_url}")
                logger.error(f"   错误: {e}")
                logger.warning(f"   跳过该模型，继续测试其他模型")
                return None

    # Create model based on framework
    try:
        if framework == "onnx" and weights_path:
            model = create_model(
                str(weights_path), device="auto", conf_threshold=conf_threshold
            )
        else:
            model = create_model(
                model_name, device="auto", conf_threshold=conf_threshold
            )
    except ValueError as e:
        logger.error(f"❌ 不支持的模型类型: {model_name}")
        logger.error(f"   错误: {e}")
        return None

    logger.info(
        f"加载模型权重: {weights_path if weights_path else '使用内置预训练权重'}"
    )

    try:
        if framework != "onnx":
            if weights_path and weights_path.exists():
                load_model_wrapper(model, str(weights_path), model_name)
            elif weights_path:
                logger.error(f"❌ 模型文件不存在: {weights_path}")
                logger.error("   请检查文件路径或先下载模型权重")
                return None
            else:
                model.load_model(None)
    except FileNotFoundError:
        if framework != "onnx":
            logger.error(f"❌ 模型文件不存在: {weights_path}")
            logger.error("   请检查文件路径或先下载模型权重")
            return None
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        logger.error(f"   模型: {model_name}")
        if framework != "onnx":
            logger.error(f"   权重文件: {weights_path}")
        return None

    model_info = model.get_model_info()
    logger.info(f"模型信息: {model_info}")

    logger.info("模型预热...")
    try:
        model.warmup()
    except Exception as e:
        logger.error(f"❌ 模型预热失败: {e}")
        logger.warning("   继续执行，但首次推理可能较慢")

    all_detections = {}
    perf_metrics = PerformanceMetrics()

    total_images = max_images if max_images else len(dataset)
    logger.info(f"将处理 {total_images} 张图片")

    image_iterator = enumerate(dataset)
    if total_images <= len(dataset):
        image_iterator = tqdm(
            image_iterator,
            total=total_images,
            desc=f"{model_name} 推理",
            unit="张",
            leave=False,
        )

    for idx, (image_id, image) in image_iterator:
        if idx >= total_images:
            break

        try:
            start_time = perf_metrics.start_timer()
            detections = model.predict(image, conf_threshold)
            inference_time = perf_metrics.end_timer(start_time)

            perf_metrics.add_inference_time(inference_time)
            all_detections[image_id] = detections

        except Exception as e:
            logger.error(
                f"❌ 推理失败 (图片 {idx}/{total_images}, ID: {image_id}): {e}"
            )
            logger.warning("   跳过此图片，继续处理下一张")
            continue

        if visualize and vis_dir and idx < num_viz_images and len(detections) > 0:
            viz_filename = f"{model_name}_vis_{idx:04d}_{image_id:012d}.jpg"
            viz_path = vis_dir / viz_filename

            class_names = model_info.get("model_yaml", {}).get("names", {})
            if not class_names and hasattr(model, "names"):
                class_names = model.names

            try:
                num_boxes = save_detection_visualization(
                    image, detections, class_names, viz_path
                )
                if idx == 0 or idx % 5 == 0:
                    logger.info(
                        f"    已保存可视化: {viz_filename} ({num_boxes} 个检测框)"
                    )
            except Exception as e:
                logger.error(f"❌ 可视化失败: {viz_filename}")
                logger.error(f"   错误: {e}")

    logger.info("生成预测结果...")
    try:
        predictions = coco_metrics_calculator.predictions_to_coco_format(all_detections)
    except Exception as e:
        logger.error(f"❌ 生成预测结果失败: {e}")
        logger.error(f"   检测数量: {len(all_detections)}")
        return None

    logger.info("计算 COCO 指标...")
    try:
        coco_metrics = coco_metrics_calculator.compute_metrics(predictions)
    except Exception as e:
        logger.error(f"❌ 计算 COCO 指标失败: {e}")
        logger.error("   请检查标注文件路径和格式")
        return None

    performance_stats = perf_metrics.compute_performance_stats()

    logger.info(f"{model_name} 指标:")
    logger.info(f"  AP@0.50: {coco_metrics['AP@0.50']:.4f}")
    logger.info(f"  AP@0.50:0.95: {coco_metrics['AP@0.50:0.95']:.4f}")
    logger.info(f"  FPS: {performance_stats['fps']:.2f}")

    result = {
        "model_name": model_name,
        "framework": framework,
        "coco_metrics": coco_metrics,
        "performance": performance_stats,
        "model_info": model_info,
    }

    return result


def main(args):
    """Main benchmark function."""
    import json

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

    logger.info("=" * 60)
    logger.info("目标检测模型性能基准测试")
    logger.info("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = None
    if args.visualize:
        vis_dir = output_dir.parent / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"可视化目录: {vis_dir}")

    dataset_config = config.get_dataset_config()
    dataset_path = dataset_config["path"]
    split = dataset_config["split"]

    logger.info(f"加载数据集: {dataset_path}/{split}")
    dataset = COCOInferenceDataset(dataset_path, split)
    logger.info(f"数据集大小: {len(dataset)} 张图片")

    annotations_file = (
        Path(dataset_path).expanduser() / "annotations" / f"instances_{split}.json"
    )
    coco_metrics_calculator = COCOMetrics(str(annotations_file))

    models_config = config.get_models_config()
    eval_config = config.get_evaluation_config()
    test_config = config.config.get("test", {})
    max_images = (
        args.num_images
        if args.num_images is not None
        else test_config.get("max_images")
    )

    conf_threshold = args.conf_threshold
    if conf_threshold is None:
        conf_threshold = eval_config.get("conf_threshold", 0.001)
        logger.info(f"使用配置文件中的置信度阈值: {conf_threshold}")
    else:
        logger.info(f"使用命令行指定的置信度阈值: {conf_threshold}")

    models_to_test = []

    # Handle ONNX format
    if args.format == "onnx":
        export_dir = Path("models_export")
        if not export_dir.exists():
            logger.error(f"ONNX 模型目录不存在: {export_dir}")
            logger.error("请先运行以下命令导出 ONNX 模型:")
            logger.error("  od-benchmark export --all-models --format onnx")
            return

        onnx_files = sorted(export_dir.glob("*.onnx"))

        if not onnx_files:
            logger.error(f"在 {export_dir} 中未找到 ONNX 模型")
            logger.error("请先运行以下命令导出 ONNX 模型:")
            logger.error("  od-benchmark export --all-models --format onnx")
            return

        if args.all:
            for onnx_file in onnx_files:
                models_to_test.append(
                    {
                        "name": onnx_file.stem,
                        "path": str(onnx_file),
                        "framework": "onnx",
                    }
                )
        elif args.model:
            for model_name in args.model:
                if model_name.lower() == "all":
                    models_to_test = models_config
                    break
                matching_files = [
                    f
                    for f in onnx_files
                    if f.stem == model_name or str(f).endswith(model_name)
                ]
                if matching_files:
                    for f in matching_files:
                        models_to_test.append(
                            {
                                "name": f.stem,
                                "path": str(f),
                                "framework": "onnx",
                            }
                        )
                else:
                    logger.warning(f"未找到 ONNX 模型: {model_name}")
        else:
            logger.error("请使用 --model <model_name> 或 --all 指定要测试的模型")
            logger.info(f"可用的 ONNX 模型: {', '.join([f.stem for f in onnx_files])}")
            return

        logger.info(f"模型格式: ONNX")
        logger.info(f"计划测试 {len(models_to_test)} 个 ONNX 模型")
    else:
        # Handle PyTorch format
        if args.all:
            models_to_test = models_config
        elif args.model:
            for model_name in args.model:
                if model_name.lower() == "all":
                    models_to_test = models_config
                    break
                for model_cfg in models_config:
                    if model_cfg["name"] == model_name:
                        models_to_test.append(model_cfg)
                        break
        else:
            logger.error("请使用 --model <model_name> 或 --all 指定要测试的模型")
            logger.info("可用的模型: " + ", ".join([m["name"] for m in models_config]))
            return

        if not models_to_test:
            logger.error("未找到要测试的模型")
            return

        logger.info(f"模型格式: PyTorch")
        logger.info(f"计划测试 {len(models_to_test)} 个模型")

    aggregator = MetricsAggregator()

    for model_config in tqdm(models_to_test, desc="模型进度", unit="模型"):
        result = run_single_model(
            model_config,
            dataset,
            coco_metrics_calculator,
            logger,
            max_images,
            conf_threshold,
            args.visualize,
            vis_dir,
            args.num_viz_images,
        )

        if result:
            aggregator.add_model_result(
                result["model_name"],
                result["coco_metrics"],
                result["performance"],
                result["model_info"],
            )

            result_file = output_dir / f"{result['model_name']}_result.json"
            try:
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info(f"结果已保存: {result_file}")
            except Exception as e:
                logger.error(f"❌ 保存结果文件失败: {result_file}")
                logger.error(f"   错误: {e}")

    logger.info("=" * 60)
    logger.info("生成汇总报告...")
    logger.info("=" * 60)

    all_results = aggregator.get_all_results()

    if not all_results:
        logger.warning("没有可用的结果用于生成报告")
        logger.info("=" * 60)
        logger.info("基准测试完成！")
        logger.info("=" * 60)
        return

    comparison_file = output_dir / "comparison.json"
    aggregator.save_results(str(comparison_file))
    logger.info(f"对比结果已保存: {comparison_file}")

    results_table = generate_results_table(all_results)
    logger.info("\n" + "=" * 60)
    logger.info("性能对比表格")
    logger.info("=" * 60)
    logger.info(results_table.to_string())

    table_file = output_dir / "results_table.csv"
    results_table.to_csv(table_file)
    logger.info(f"表格已保存: {table_file}")

    figures_dir = output_dir.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_metrics_comparison(
            all_results,
            ["AP@0.50", "AP@0.50:0.95", "fps"],
            str(figures_dir / "metrics_comparison.png"),
        )
        logger.info(f"指标对比图已保存: {figures_dir / 'metrics_comparison.png'}")
    except Exception as e:
        logger.error(f"❌ 生成指标对比图失败: {e}")

    try:
        plot_fps_vs_map(all_results, str(figures_dir / "fps_vs_map.png"))
        logger.info(f"FPS vs mAP 图已保存: {figures_dir / 'fps_vs_map.png'}")
    except Exception as e:
        logger.error(f"❌ 生成 FPS vs mAP 图失败: {e}")

    try:
        plot_model_size_vs_performance(
            all_results, str(figures_dir / "size_vs_performance.png")
        )
        logger.info(
            f"模型大小 vs 性能图已保存: {figures_dir / 'size_vs_performance.png'}"
        )
    except Exception as e:
        logger.error(f"❌ 生成模型大小 vs 性能图失败: {e}")

    logger.info("=" * 60)
    logger.info("基准测试完成！")
    logger.info("=" * 60)
