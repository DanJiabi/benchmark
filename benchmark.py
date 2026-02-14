#!/usr/bin/env python3
"""
Benchmark - Object Detection Performance Evaluation Tool

This is kept for backward compatibility and is the primary entry point.
"""

import sys
from pathlib import Path

# Ensure we're in the project directory
project_root = Path(__file__).resolve().parent
if project_root.name in ["Scripts", "Scripts", "scripts"]:
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

import argparse

parser = argparse.ArgumentParser(
    description="目标检测模型性能基准测试",
)

parser.add_argument(
    "--config",
    type=str,
    default="config.yaml",
    help="配置文件路径（默认: config.yaml）",
)
parser.add_argument(
    "--model",
    type=str,
    action="append",
    help="指定要测试的模型",
)
parser.add_argument(
    "--all",
    action="store_true",
    help="测试所有配置的模型",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="outputs/results",
    help="输出目录（默认: outputs/results）",
)
parser.add_argument(
    "--visualize",
    action="store_true",
    help="保存检测框可视化图片",
)
parser.add_argument(
    "--num-viz-images",
    type=int,
    default=10,
    help="可视化图片数量（默认: 10）",
)
parser.add_argument(
    "--conf-threshold",
    type=float,
    default=None,
    help="置信度阈值（默认: 使用配置文件中的值）",
)
parser.add_argument(
    "--num-images",
    type=int,
    default=None,
    help="测试图片数量（默认: 全部数据）",
)

args = parser.parse_args()

from src.utils.logger import Config, setup_logger

try:
    config = Config(args.config)
    logger = setup_logger(config)
except FileNotFoundError:
    print(f"❌ 配置文件不存在: {args.config}")
    print("   请检查配置文件路径")
    sys.exit(1)
except Exception as e:
    print(f"❌ 配置文件加载失败: {e}")
    print(f"   文件: {args.config}")
    sys.exit(1)

logger.info("=" * 60)
logger.info("目标检测模型性能基准测试")
logger.info("=" * 60)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

dataset_config = config.get_dataset_config()
dataset_path = dataset_config["path"]
split = dataset_config["split"]

logger.info(f"加载数据集: {dataset_path}/{split}")
logger.info(f"数据集大小: {len(COCOInferenceDataset(dataset_path, split)) 张图片")

models_config = config.get_models_config()
eval_config = config.get_evaluation_config()
conf_threshold = args.conf_threshold or eval_config.get("conf_threshold", 0.001)
test_config = config.config.get("test", {})
max_images = args.num_images or test_config.get("max_images")

models_to_test = []
if args.all:
    models_to_test = models_config
elif args.model:
    for model_name in args.model:
        for model_cfg in models_config:
            if model_cfg["name"] == model_name:
                models_to_test.append(model_cfg)
                break
else:
    logger.error("请使用 --model <model_name> 或 --all 指定要测试的模型")
    logger.info(f"可用的模型: {', '.join([m['name'] for m in models_config])}")
    sys.exit(1)

if not models_to_test:
    logger.error("未找到要测试的模型")
    logger.info(f"可用的模型: {', '.join([m['name'] for m in models_config])}")
    sys.exit(1)

logger.info(f"计划测试 {len(models_to_test)} 个模型")

from src.models import create_model, load_model_wrapper
from src.data.coco_dataset import COCOInferenceDataset
from src.metrics.coco_metrics import COCOMetrics, PerformanceMetrics, MetricsAggregator
from src.utils.visualization import (
    save_detection_visualization,
    plot_metrics_comparison,
    plot_fps_vs_map,
    plot_model_size_vs_performance,
    generate_results_table,
)


def run_single_model(
    model_config: dict,
    dataset: COCOInferenceDataset,
    annotations_file: str,
    coco_metrics_calculator: COCOMetrics,
    logger,
    max_images: int = None,
    conf_threshold: float = 0.001,
    visualize: bool = False,
    vis_dir: Path = None,
    num_viz_images: int = 10,
    model_info: dict = None,
):
    model_name = model_config["name"]
    framework = model_config["framework"]
    weights_file = model_config["weights"]
    weights_url = model_config.get("url")

    logger.info(f"开始评估模型: {model_name}")

    try:
        model = create_model(model_name, device="auto", conf_threshold=conf_threshold)
    except ValueError as e:
        logger.error(f"❌ 不支持的模型类型: {model_name}")
        logger.error(f"   错误: {e}")
        return None

    weights_path = None
    if weights_file:
        weights_path = Path("models_cache") / weights_file
        if weights_url and not weights_path.exists():
            logger.info(f"下载模型权重: {weights_url}")
            try:
                from src.utils.logger import download_model_weights
                download_model_weights(weights_url, weights_path)
            except Exception as e:
                logger.error(f"❌ 模型下载失败: {model_name}")
                logger.error(f"   URL: {weights_url}")
                logger.error(f"   错误: {e}")
                logger.warning(f"   跳过该模型，继续测试其他模型")
                return None

    logger.info(
        f"加载模型权重: {weights_path if weights_path else '使用内置预训练权重'}"
    )

    try:
        if weights_path:
            load_model_wrapper(model, str(weights_path), model_name)
        else:
            model.load_model(None)
    except FileNotFoundError:
        logger.error(f"❌ 模型文件不存在: {weights_path}")
        logger.error("   请检查文件路径或先下载模型权重")
        return None
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        logger.error(f"   模型: {model_name}")
        logger.error(f"   权重文件: {weights_path}")
        return None

    if model_info is None:
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

    total_images = max_images
    logger.info(f"将处理 {total_images} 张图片")

    for idx, (image_id, image) in enumerate(dataset):
        if idx >= total_images:
            break

        try:
            start_time = perf_metrics.start_timer()
            detections = model.predict(image, conf_threshold)
            inference_time = perf_metrics.end_timer(start_time)

            perf_metrics.add_inference_time(inference_time)
            all_detections[image_id] = detections

        except Exception as e:
            logger.warning(f"推理失败 (图片 {idx}/{total_images}, ID: {image_id}): {e}")
            continue

        if visualize and vis_dir and idx < num_viz_images:
            viz_filename = (
                f"{model_name}_vis_{idx:04d}_{image_id:012d}.jpg"
            )
            viz_path = vis_dir / viz_filename

            class_names = model_info.get("model_yaml", {}).get("names", {})
            if not class_names:
                try:
                    class_names = model_info.get("names", {})
                except Exception:
                    pass

            try:
                num_boxes = save_detection_visualization(
                    image, detections, class_names, viz_path, max_boxes=10
                )
                if idx == 0 or idx % 20 == 0:
                    logger.info(f"    已保存可视化: {viz_filename} ({num_boxes} 个检测框)")
            except Exception as e:
                logger.warning(f"    可视化失败: {viz_filename}")
                logger.warning(f"    错误: {e}")

    logger.info(f"生成预测结果...")

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

    return {
        "model_name": model_name,
        "framework": framework,
        "coco_metrics": coco_metrics,
        "performance": performance_stats,
        "model_info": model_info,
        "num_images": total_images,
        "num_detections": len(all_detections),
    }


def benchmark_main():
    """Main benchmark function"""
    from src.utils.logger import Config, setup_logger

    args = sys.argv[1:]

    parser = argparse.ArgumentParser(description="目标检测模型性能基准测试")

    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--model", type=str, action="append", help="指定要测试的模型")
    parser.add_argument("--all", action="store_true", help="测试所有配置的模型")
    parser.add_argument("--output-dir", type=str, default="outputs/results", help="输出目录")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="保存检测框可视化图片",
    )
    parser.add_argument(
        "--num-viz-images",
        type=int,
        default=10,
        help="可视化图片数量（默认: 10）",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="置信度阈值（默认: 使用配置文件中的值）",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="测试图片数量（默认: 全部数据）",
    )

    args = parser.parse_args()

    try:
        config = Config(args.config)
        logger = setup_logger(config)
    except FileNotFoundError:
        print(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        print(f"   文件: {args.config}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("目标检测模型性能基准测试")
    logger.info("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = config.get_dataset_config()
    dataset_path = dataset_config["path"]
    split = dataset_config["split"]

    logger.info(f"加载数据集: {dataset_path}/{split}")
    logger.info(f"数据集大小: {len(COCOInferenceDataset(dataset_path, split)) 张图片")

    annotations_file = (
        Path(dataset_path).expanduser() / "annotations" / f"instances_{split}.json"
    )
    coco_metrics_calculator = COCOMetrics(str(annotations_file))

    models_config = config.get_models_config()
    eval_config = config.get_evaluation_config()
    test_config = config.config.get("test", {})
    max_images = args.num_images or test_config.get("max_images")

    conf_threshold = args.conf_threshold or eval_config.get("conf_threshold", 0.001)

    models_to_test = []
    if args.all:
        models_to_test = models_config
    elif args.model:
        for model_name in args.model:
            for model_cfg in models_config:
                if model_cfg["name"] == model_name:
                    models_to_test.append(model_cfg)
                    break
    else:
        logger.error(f"未找到模型: {model_name}")
                logger.info(f"可用的模型: {', '.join([m['name'] for m in models_config])}")
                sys.exit(1)
    else:
        logger.error("请使用 --model <model_name> 或 --all 指定要测试的模型")
        logger.info(f"可用的模型: {', '.join([m['name'] for m in models_config])}")
        sys.exit(1)

    if not models_to_test:
        logger.error("未找到要测试的模型")
        logger.info(f"可用的模型: {', '.join([m['name'] for m in models_config])}")
        sys.exit(1)

    logger.info(f"计划测试 {len(models_to_test)} 个模型")

    from src.models import create_model, load_model_wrapper
    from src.data.coco_dataset import COCOInferenceDataset
    from src.metrics.coco_metrics import COCOMetrics, PerformanceMetrics, MetricsAggregator
    from src.utils.visualization import (
        save_detection_visualization,
        plot_metrics_comparison,
        plot_fps_vs_map,
        plot_model_size_vs_performance,
        generate_results_table,
    )

    aggregator = MetricsAggregator()

    for model_config in models_to_test:
        result = run_single_model(
            model_config,
            COCOInferenceDataset(dataset_path, split),
            annotations_file,
            COCOMetrics(annotations_file),
            None,
            max_images,
            conf_threshold,
            args.visualize,
            output_dir / "visualizations" if args.visualize else None,
            num_viz_images=args.num_viz_images if args.visualize else None,
            model_info=None,
        )

        if result:
            aggregator.add_model_result(
                result["model_name"],
                result["coco_metrics"],
                result["performance"],
                result["model_info"],
            )

            import json

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
        logger.info(f"模型大小 vs 性能图已保存: {figures_dir / 'size_vs_performance.png'}")
    except Exception as e:
        logger.error(f"❌ 生成模型大小 vs 性能图失败: {e}")

    logger.info("=" * 60)
    logger.info("基准测试完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    benchmark_main()
