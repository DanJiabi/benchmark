"""
Command line interface for od-benchmark package
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

import numpy as np
from tqdm import tqdm

from src.models.base import Detection
from src.models import (
    create_model,
    load_model_wrapper,
    UserModelLoader,
    export_model_cli,
    batch_export_models,
)
from src.data.coco_dataset import COCOInferenceDataset
from src.metrics.coco_metrics import COCOMetrics, PerformanceMetrics, MetricsAggregator
from src.utils.logger import Config, setup_logger, download_model_weights
from src.utils.visualization import (
    save_detection_visualization,
    plot_metrics_comparison,
    plot_fps_vs_map,
    plot_model_size_vs_performance,
    generate_results_table,
)
from src.analysis import ModelComparison


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


def benchmark_main():
    """Main benchmark function"""
    parser = argparse.ArgumentParser(description="目标检测模型性能基准测试")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="配置文件路径"
    )
    parser.add_argument("--model", type=str, action="append", help="指定要测试的模型")
    parser.add_argument("--all", action="store_true", help="测试所有配置的模型")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/results", help="输出目录"
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


def analyze_main(args=None):
    """Main analyze function"""
    # 如果 args 为 None，则解析参数（用于直接调用）
    if args is None:
        parser = argparse.ArgumentParser(
            description="Object Detection Benchmark - Model Analysis",
            epilog="Run 'od-benchmark analyze --help' for more information.",
        )

        parser.add_argument(
            "--baseline",
            type=str,
            action="append",
            help="Baseline model name(s) from config.yaml",
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
            required=True,
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

        args = parser.parse_args()

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

    # 加载配置获取模型
    models_config = config.get_models_config()
    dataset_config = config.get_dataset_config()
    dataset_path = dataset_config["path"]
    split = dataset_config["split"]

    logger.info(f"加载数据集: {dataset_path}/{split}")
    dataset = COCOInferenceDataset(dataset_path, split)
    logger.info(f"数据集大小: {len(dataset)} 张图片")

    # 获取基准模型列表
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

    # 获取用户模型列表
    user_models = args.user_model if args.user_model else []
    logger.info(f"用户模型: {user_models}")

    conf_threshold = 0.001

    # 获取标注文件
    annotations_file = (
        Path(dataset_path).expanduser() / "annotations" / f"instances_{split}.json"
    )

    # 运行所有对比
    from src.analysis import ModelComparison

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
                from . import load_model_wrapper

                load_model_wrapper(
                    baseline_model,
                    str(Path("models_cache") / weights_file),
                    baseline_name,
                )
        except ValueError as e:
            logger.error(f"❌ 基准模型 {baseline_name} 加载失败: {e}")
            continue

        # 对每个用户模型进行对比
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

    # 保存汇总结果
    summary = {
        "timestamp": datetime.now().isoformat(),
        "num_comparisons": len(all_comparisons),
        "baseline_models": [m["name"] for m in baseline_models],
        "user_models": user_models,
        "comparisons": all_comparisons,
    }

    # 保存所有结果
    formats = []
    if args.format == "all":
        formats = ["json", "html", "csv"]
    else:
        formats = [args.format]

    # 保存汇总
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"汇总结果已保存: {summary_file}")

    # 为每个对比保存单独结果
    for idx, comp in enumerate(all_comparisons):
        comp_dir = output_dir / f"comparison_{idx:03d}"
        comp_dir.mkdir(parents=True, exist_ok=True)

        comp_file = comp_dir / "comparison.json"
        with open(comp_file, "w", encoding="utf-8") as f:
            json.dump(comp, f, indent=2, ensure_ascii=False)

        # HTML
        if "html" in formats:
            html_content = _generate_multi_model_html_report(comp)
            html_file = comp_dir / "comparison.html"
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html_content)

        # CSV
        if "csv" in formats:
            csv_content = _generate_multi_model_csv_report(comp)
            csv_file = comp_dir / "comparison.csv"
            with open(csv_file, "w", encoding="utf-8") as f:
                f.write(csv_content)

    logger.info("")
    logger.info("=" * 70)
    logger.info("所有对比分析完成！")
    logger.info(f"汇总: {summary_file}")
    logger.info(f"对比结果: {output_dir}")
    logger.info("=" * 70)


def _generate_multi_model_html_report(comparison: dict) -> str:
    """生成多模型对比 HTML 报告"""
    if not comparison:
        return ""

    baseline = comparison.get("baseline_results", {})
    user = comparison.get("user_results", {})
    comp = comparison.get("comparison", {})

    html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型对比分析报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        .header {{
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .info-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .metrics-table th {{
            background: #343a40;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .metrics-table td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        .recommendations {{
            background: #fff3cd;
            padding: 20px;
            border-radius: 6px;
            margin: 20px 0;
        }}
        .recommendation-item {{
            padding: 8px 0;
            border-bottom: 1px solid #ffeaa7;
        }}
        .recommendation-item:last-child {{
            border-bottom: none;
        }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
        .timestamp {{
            color: #6c757d;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 模型对比分析报告</h1>
            <p class="timestamp">生成时间: {comparison.get("timestamp", "N/A")}</p>
        </div>

        <div class="info-grid">
            <div class="info-card">
                <h3>基准模型</h3>
                <p><strong>名称:</strong> {comparison.get("baseline_name", "N/A")}</p>
            </div>
            <div class="info-card">
                <h3>用户模型</h3>
                <p><strong>标识:</strong> {comparison.get("user_model_spec", "N/A")}</p>
            </div>
        </div>

        <h2>📈 对比结果</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>指标</th>
                    <th>基准模型</th>
                    <th>用户模型</th>
                    <th>差异</th>
                    <th>变化 %</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>mAP@0.50</strong></td>
                    <td>{baseline.get("mAP@0.50", 0):.4f}</td>
                    <td>{user.get("mAP@0.50", 0):.4f}</td>
                    <td class="{"positive" if comp.get("map_diff", 0) > 0 else "negative"}">{comp.get("map_diff", 0):+.4f}</td>
                    <td class="{"positive" if comp.get("map_diff_pct", 0) > 0 else "negative"}">{comp.get("map_diff_pct", 0):+.2f}%</td>
                </tr>
                <tr>
                    <td><strong>mAP@0.50:0.95</strong></td>
                    <td>{baseline.get("mAP@0.50:0.95", 0):.4f}</td>
                    <td>{user.get("mAP@0.50:0.95", 0):.4f}</td>
                    <td class="{"positive" if comp.get("map_diff_95", 0) > 0 else "negative"}">{comp.get("map_diff_95", 0):+.4f}</td>
                    <td class="{"positive" if comp.get("map_diff_95_pct", 0) > 0 else "negative"}">{comp.get("map_diff_95_pct", 0):+.2f}%</td>
                </tr>
            </tbody>
        </table>

        <h2>💡 建议</h2>
        <div class="recommendations">
            <div class="recommendation-item">
                {comparison.get("recommendation", "N/A")}
            </div>
        </div>

        <p style="text-align: center; color: #6c757d; font-style: italic; margin-top: 40px;">
            报告由 od-benchmark 生成
        </p>
    </div>
</body>
</html>
"""
    return html


def _generate_multi_model_csv_report(comparison: dict) -> str:
    """生成多模型对比 CSV 报告"""
    if not comparison:
        return ""

    comp = comparison.get("comparison", {})

    lines = []
    lines.append("指标,基准模型,用户模型,差异,变化%")
    lines.append("-" * 50)

    lines.append(
        f"mAP@0.50,{comp.get('baseline_map_50', 0):.4f},{comp.get('user_map_50', 0):.4f},{comp.get('map_diff', 0):+.4f},{comp.get('map_diff_pct', 0):+.2f}%"
    )
    lines.append(
        f"mAP@0.50:0.95,{comp.get('baseline_map_95', 0):.4f},{comp.get('user_map_95', 0):.4f},{comp.get('map_diff_95', 0):+.4f},{comp.get('map_diff_95_pct', 0):+.2f}%"
    )
    lines.append("")
    lines.append(f"建议,{comparison.get('recommendation', 'N/A')}")

    return "\n".join(lines)
    logger.info("模型对比分析")
    logger.info("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载配置获取基准模型
    models_config = config.get_models_config()
    dataset_config = config.get_dataset_config()
    dataset_path = dataset_config["path"]
    split = dataset_config["split"]

    logger.info(f"加载数据集: {dataset_path}/{split}")
    dataset = COCOInferenceDataset(dataset_path, split)
    logger.info(f"数据集大小: {len(dataset)} 张图片")

    # 查找基准模型
    baseline_config = None
    for model_cfg in models_config:
        if model_cfg["name"] == args.baseline:
            baseline_config = model_cfg
            break

    if not baseline_config:
        logger.error(f"❌ 未找到基准模型: {args.baseline}")
        logger.info(f"可用的模型: {', '.join([m['name'] for m in models_config])}")
        return

    logger.info(f"基准模型: {args.baseline}")
    logger.info(f"用户模型: {args.user_model}")

    # 加载模型
    conf_threshold = 0.001

    try:
        baseline_model = create_model(
            args.baseline, device="auto", conf_threshold=conf_threshold
        )
        weights_file = baseline_config.get("weights")
        if weights_file:
            from . import load_model_wrapper

            load_model_wrapper(
                baseline_model, str(Path("models_cache") / weights_file), args.baseline
            )
    except ValueError as e:
        logger.error(f"❌ 基准模型加载失败: {e}")
        return

    # 加载用户模型
    try:
        user_model = UserModelLoader.load_user_model(
            args.user_model, device="auto", conf_threshold=conf_threshold
        )
    except Exception as e:
        logger.error(f"❌ 用户模型加载失败: {e}")
        return

    # 创建对比器
    comparison = ModelComparison(baseline_model, user_model, logger)

    # 获取标注文件
    annotations_file = (
        Path(dataset_path).expanduser() / "annotations" / f"instances_{split}.json"
    )

    # 运行对比
    comparison.run_comparison(
        dataset=dataset,
        annotations_file=str(annotations_file),
        max_images=args.num_images,
        conf_threshold=conf_threshold,
    )

    # 保存结果
    formats = []
    if args.format == "all":
        formats = ["json", "html", "csv"]
    else:
        formats = [args.format]

    comparison.save_results(output_dir, formats)

    logger.info("")
    logger.info("=" * 70)
    logger.info("分析完成！")
    logger.info(f"结果已保存到: {output_dir}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Object Detection Benchmark - Performance evaluation tool",
        epilog="Run 'od-benchmark <command> --help' for more information on a command.",
    )

    parser.add_argument("--version", action="version", version="od-benchmark 0.1.0")

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=False
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run benchmark evaluation",
        description="Run object detection model benchmark evaluation",
    )
    benchmark_parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path (default: config.yaml)",
    )
    benchmark_parser.add_argument(
        "--model",
        type=str,
        action="append",
        help="Specify model(s) to test (can be used multiple times)",
    )
    benchmark_parser.add_argument(
        "--all", action="store_true", help="Test all configured models"
    )
    benchmark_parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Number of test images (default: all data)",
    )
    benchmark_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable detection box visualization",
    )
    benchmark_parser.add_argument(
        "--num-viz-images",
        type=int,
        default=10,
        help="Number of visualization images (default: 10)",
    )
    benchmark_parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="Confidence threshold (default: use config file value)",
    )
    benchmark_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/results",
        help="Output directory (default: outputs/results)",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Compare baseline model with user model",
        description="Compare baseline model performance with user custom model",
    )
    analyze_parser.add_argument(
        "--baseline",
        type=str,
        action="append",
        help="Baseline model name(s) from config.yaml (can be used multiple times)",
    )
    analyze_parser.add_argument(
        "--all-baselines",
        action="store_true",
        help="Use all configured baseline models",
    )
    analyze_parser.add_argument(
        "--user-model",
        type=str,
        action="append",
        help="User model(s)",
    )
    analyze_parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path",
    )
    analyze_parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="Number of test images",
    )
    analyze_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/analysis",
        help="Output directory",
    )
    analyze_parser.add_argument(
        "--format",
        type=str,
        default="all",
        choices=["json", "html", "csv", "all"],
        help="Output format",
    )
    analyze_parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )

    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export model to ONNX or TensorRT format",
        description="Export YOLO models to ONNX or TensorRT format for optimized inference",
    )
    export_parser.add_argument(
        "--model",
        type=str,
        action="append",
        help="Path to model weights file(s) (.pt), can be used multiple times",
    )
    export_parser.add_argument(
        "--all-models",
        action="store_true",
        help="Export all models from models_cache directory",
    )
    export_parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "tensorrt", "all"],
        help="Export format (default: onnx)",
    )
    export_parser.add_argument(
        "--output-dir",
        type=str,
        default="models_export",
        help="Output directory (default: models_export)",
    )
    export_parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("H", "W"),
        help="Input image size (default: 640 640)",
    )
    export_parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Use dynamic input size (ONNX only)",
    )
    export_parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Simplify ONNX model (default: True)",
    )
    export_parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 precision (TensorRT only, default: True)",
    )
    export_parser.add_argument(
        "--int8",
        action="store_true",
        help="Use INT8 quantization (TensorRT only)",
    )
    export_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for export (default: 1)",
    )
    export_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for export (default: cpu)",
    )

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare model performance across different formats (PyTorch vs ONNX)",
        description="Compare model performance across different formats",
    )
    compare_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model weights file (.pt)",
    )
    compare_parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name (default: auto-detect from filename)",
    )
    compare_parser.add_argument(
        "--formats",
        type=str,
        default="pytorch,onnx",
        help="Formats to compare, comma-separated (default: pytorch,onnx)",
    )
    compare_parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file path (default: config.yaml)",
    )
    compare_parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="Number of test images (default: 50)",
    )
    compare_parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/format_comparison",
        help="Output directory (default: outputs/format_comparison)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "benchmark":
        benchmark_main()

    elif args.command == "analyze":
        analyze_main(args)

    elif args.command == "export":
        # 参数验证：必须指定 --model 或 --all-models
        if not args.model and not args.all_models:
            print("错误: 必须指定 --model 或 --all-models")
            print("示例:")
            print("  od-benchmark export --model model.pt")
            print("  od-benchmark export --model model1.pt --model model2.pt")
            print("  od-benchmark export --all-models")
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
        )

    elif args.command == "compare":
        from src.analysis import compare_model_formats_cli

        formats = [f.strip() for f in args.formats.split(",")]
        compare_model_formats_cli(
            model_path=args.model,
            model_name=args.model_name,
            formats=formats,
            config=args.config,
            num_images=args.num_images,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()
        return


if __name__ == "__main__":
    main()
