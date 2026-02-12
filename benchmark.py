import argparse
from pathlib import Path
from typing import Dict, List, Any

from src.models.base import BaseModel
from src.models.yolov8 import YOLOv8, YOLOv11
from src.models.yolov9 import YOLOv9, YOLOv10
from src.models.rt_detr import RTDETR
from src.models.faster_rcnn import FasterRCNN
from src.data.coco_dataset import COCOInferenceDataset
from src.metrics.coco_metrics import COCOMetrics, PerformanceMetrics, MetricsAggregator
from src.utils.logger import Config, setup_logger, download_model_weights
from src.utils.visualization import (
    plot_metrics_comparison,
    plot_fps_vs_map,
    plot_model_size_vs_performance,
    generate_results_table,
    save_results_table,
)


MODEL_REGISTRY = {
    "yolov8": YOLOv8,
    "yolov9": YOLOv9,
    "yolov10": YOLOv10,
    "yolov11": YOLOv11,
    "rtdetr": RTDETR,
    "faster_rcnn": FasterRCNN,
}


def run_single_model(
    model_config: Dict[str, Any],
    dataset: COCOInferenceDataset,
    coco_metrics_calculator: COCOMetrics,
    logger,
    max_images: int = None,
) -> Dict[str, Any]:
    model_name = model_config["name"]
    framework = model_config["framework"]
    weights_file = model_config["weights"]
    weights_url = model_config.get("url")

    logger.info(f"开始评估模型: {model_name}")

    model_type_key = None
    for key in MODEL_REGISTRY.keys():
        if model_name.lower().startswith(key):
            model_type_key = key
            break

    if model_type_key is None:
        logger.error(f"不支持的模型类型: {model_name}")
        return None

    model_class = MODEL_REGISTRY.get(model_type_key)

    model = model_class(device="auto")

    weights_path = Path("models_cache") / weights_file
    if weights_url and not weights_path.exists():
        logger.info(f"下载模型权重: {weights_url}")
        download_model_weights(weights_url, weights_path)

    logger.info(f"加载模型权重: {weights_path}")
    model.load_model(str(weights_path))

    model_info = model.get_model_info()
    logger.info(f"模型信息: {model_info}")

    logger.info("模型预热...")
    model.warmup()

    all_detections = {}
    perf_metrics = PerformanceMetrics()

    total_images = max_images if max_images else len(dataset)
    logger.info(f"将处理 {total_images} 张图片")

    for idx, (image_id, image) in enumerate(dataset):
        if idx >= total_images:
            break

        if idx % 100 == 0:
            logger.info(f"处理进度: {idx}/{total_images}")

        start_time = perf_metrics.start_timer()
        detections = model.predict(image)
        inference_time = perf_metrics.end_timer(start_time)

        perf_metrics.add_inference_time(inference_time)
        all_detections[image_id] = detections

    logger.info("生成预测结果...")
    predictions = coco_metrics_calculator.predictions_to_coco_format(all_detections)

    logger.info("计算 COCO 指标...")
    coco_metrics = coco_metrics_calculator.compute_metrics(predictions)

    performance_stats = perf_metrics.compute_performance_stats()

    logger.info(f"{model_name} 指标:")
    logger.info(f"  mAP@0.5: {coco_metrics['mAP50']:.4f}")
    logger.info(f"  mAP@0.5:0.95: {coco_metrics['mAP50-95']:.4f}")
    logger.info(f"  FPS: {performance_stats['fps']:.2f}")

    result = {
        "model_name": model_name,
        "framework": framework,
        "coco_metrics": coco_metrics,
        "performance": performance_stats,
        "model_info": model_info,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="目标检测模型性能基准测试")
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="配置文件路径"
    )
    parser.add_argument("--model", type=str, action="append", help="指定要测试的模型")
    parser.add_argument("--all", action="store_true", help="测试所有配置的模型")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/results", help="输出目录"
    )

    args = parser.parse_args()

    config = Config(args.config)
    logger = setup_logger(config)

    logger.info("=" * 60)
    logger.info("目标检测模型性能基准测试")
    logger.info("=" * 60)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    max_images = test_config.get("max_images")

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

    for model_config in models_to_test:
        result = run_single_model(
            model_config, dataset, coco_metrics_calculator, logger, max_images
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
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"结果已保存: {result_file}")

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

    plot_metrics_comparison(
        all_results,
        ["mAP50", "mAP50-95", "fps"],
        str(figures_dir / "metrics_comparison.png"),
    )
    logger.info(f"指标对比图已保存: {figures_dir / 'metrics_comparison.png'}")

    plot_fps_vs_map(all_results, str(figures_dir / "fps_vs_map.png"))
    logger.info(f"FPS vs mAP 图已保存: {figures_dir / 'fps_vs_map.png'}")

    plot_model_size_vs_performance(
        all_results, str(figures_dir / "size_vs_performance.png")
    )
    logger.info(f"模型大小 vs 性能图已保存: {figures_dir / 'size_vs_performance.png'}")

    logger.info("=" * 60)
    logger.info("基准测试完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
