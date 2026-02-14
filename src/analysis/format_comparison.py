"""
格式对比分析模块

比较 PyTorch、ONNX 等不同格式的性能差异
"""

from typing import Dict, List, Any
from pathlib import Path
import json
from datetime import datetime
import numpy as np

from src.models.base import BaseModel
from src.models import create_model, ONNXExporter
from src.data.coco_dataset import COCOInferenceDataset
from src.metrics.coco_metrics import COCOMetrics, PerformanceMetrics
from src.analysis import ModelComparison


class FormatComparison:
    """格式性能对比分析器"""

    SUPPORTED_FORMATS = ["pytorch", "onnx"]

    def __init__(
        self,
        model_name: str,
        model_path: str,
        output_dir: str = "outputs/format_comparison",
        logger=None,
    ):
        """
        初始化格式对比分析器

        Args:
            model_name: 模型名称
            model_path: 原始模型路径 (.pt)
            output_dir: 输出目录
            logger: 日志记录器
        """
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

        self.results = {}

    def compare_formats(
        self,
        dataset: COCOInferenceDataset,
        annotations_file: str,
        formats: List[str] = None,
        max_images: int = 50,
        conf_threshold: float = 0.001,
    ) -> Dict[str, Any]:
        """
        对比不同格式的性能

        Args:
            dataset: 数据集
            annotations_file: 标注文件路径
            formats: 要对比的格式列表 ["pytorch", "onnx"]
            max_images: 最大测试图片数
            conf_threshold: 置信度阈值

        Returns:
            对比结果
        """
        if formats is None:
            formats = ["pytorch", "onnx"]

        if self.logger:
            self.logger.info("=" * 70)
            self.logger.info("格式性能对比分析")
            self.logger.info("=" * 70)
            self.logger.info(f"模型: {self.model_name}")
            self.logger.info(f"对比格式: {formats}")
            self.logger.info(f"测试图片数: {max_images}")

        # 确保 ONNX 模型已导出
        if "onnx" in formats:
            self._ensure_onnx_exported()

        # 对比每种格式
        for fmt in formats:
            if self.logger:
                self.logger.info(f"\n测试 {fmt.upper()} 格式...")

            try:
                result = self._evaluate_format(
                    fmt=fmt,
                    dataset=dataset,
                    annotations_file=annotations_file,
                    max_images=max_images,
                    conf_threshold=conf_threshold,
                )
                self.results[fmt] = result

                if self.logger:
                    self.logger.info(f"✅ {fmt.upper()} 测试完成")
                    self.logger.info(f"   FPS: {result['performance']['fps']:.2f}")
                    self.logger.info(
                        f"   mAP@0.50: {result['coco_metrics']['AP@0.50']:.4f}"
                    )

            except Exception as e:
                if self.logger:
                    self.logger.error(f"❌ {fmt.upper()} 测试失败: {e}")
                self.results[fmt] = {"error": str(e), "success": False}

        # 生成对比报告
        comparison = self._generate_comparison()

        if self.logger:
            self._print_summary(comparison)

        return comparison

    def _ensure_onnx_exported(self):
        """确保 ONNX 模型已导出"""
        onnx_path = self.output_dir / f"{self.model_path.stem}.onnx"

        if not onnx_path.exists():
            if self.logger:
                self.logger.info(f"导出 ONNX 模型: {onnx_path}")

            from src.models.exporters import ONNXExporter

            exporter = ONNXExporter(
                str(self.model_path),
                str(self.output_dir),
                input_size=(640, 640),
            )
            result = exporter.export()

            if not result.get("success"):
                raise RuntimeError(f"ONNX 导出失败: {result.get('error')}")

    def _evaluate_format(
        self,
        fmt: str,
        dataset: COCOInferenceDataset,
        annotations_file: str,
        max_images: int,
        conf_threshold: float,
    ) -> Dict[str, Any]:
        """评估特定格式的性能"""
        # 加载模型
        if fmt == "pytorch":
            model = create_model(
                self.model_name, device="auto", conf_threshold=conf_threshold
            )
            from src.models import load_model_wrapper

            load_model_wrapper(model, str(self.model_path), self.model_name)
        elif fmt == "onnx":
            onnx_path = self.output_dir / f"{self.model_path.stem}.onnx"
            model = create_model(
                str(onnx_path), device="auto", conf_threshold=conf_threshold
            )
        else:
            raise ValueError(f"不支持的格式: {fmt}")

        # 预热
        model.warmup()

        # 创建 COCO 指标计算器
        coco_metrics = COCOMetrics(annotations_file)

        # 推理
        all_detections = {}
        perf_metrics = PerformanceMetrics()

        for idx, (image_id, image) in enumerate(dataset):
            if idx >= max_images:
                break

            start_time = perf_metrics.start_timer()
            detections = model.predict(image, conf_threshold)
            inference_time = perf_metrics.end_timer(start_time)

            perf_metrics.add_inference_time(inference_time)
            all_detections[image_id] = detections

        # 计算指标
        predictions = coco_metrics.predictions_to_coco_format(all_detections)
        coco_results = coco_metrics.compute_metrics(predictions)
        performance = perf_metrics.compute_performance_stats()

        return {
            "format": fmt,
            "model_name": self.model_name,
            "coco_metrics": coco_results,
            "performance": performance,
            "num_images": len(all_detections),
            "success": True,
        }

    def _generate_comparison(self) -> Dict[str, Any]:
        """生成格式对比报告"""
        comparison = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "formats": {},
            "speed_comparison": {},
            "accuracy_comparison": {},
            "recommendations": [],
        }

        # 收集各格式结果
        for fmt, result in self.results.items():
            if result.get("success"):
                perf = result.get("performance", {})
                coco = result.get("coco_metrics", {})
                comparison["formats"][fmt] = {
                    "fps": perf.get("fps", 0),
                    "avg_time_ms": perf.get("avg_time_ms", 0),
                    "map_50": coco.get("AP@0.50", 0),
                    "map_50_95": coco.get("AP@0.50:0.95", 0),
                }

        # 速度对比
        if len(comparison["formats"]) >= 2:
            pytorch_fps = comparison["formats"].get("pytorch", {}).get("fps", 0)
            onnx_fps = comparison["formats"].get("onnx", {}).get("fps", 0)

            if pytorch_fps > 0 and onnx_fps > 0:
                speedup = onnx_fps / pytorch_fps
                speedup_pct = (onnx_fps - pytorch_fps) / pytorch_fps * 100

                comparison["speed_comparison"] = {
                    "pytorch_fps": pytorch_fps,
                    "onnx_fps": onnx_fps,
                    "speedup": speedup,
                    "speedup_pct": speedup_pct,
                }

                if speedup_pct > 10:
                    comparison["recommendations"].append(
                        f"✅ ONNX 比 PyTorch 快 {speedup_pct:.1f}%，建议使用 ONNX 部署"
                    )
                elif speedup_pct > -10:
                    comparison["recommendations"].append(
                        f"⚖️ ONNX 和 PyTorch 性能相近，可根据部署环境选择"
                    )
                else:
                    comparison["recommendations"].append(
                        f"⚠️ PyTorch 比 ONNX 快 {abs(speedup_pct):.1f}%，建议继续使用 PyTorch"
                    )

        # 精度对比
        if len(comparison["formats"]) >= 2:
            pytorch_map = comparison["formats"].get("pytorch", {}).get("map_50", 0)
            onnx_map = comparison["formats"].get("onnx", {}).get("map_50", 0)

            if pytorch_map > 0 and onnx_map > 0:
                map_diff = abs(onnx_map - pytorch_map)

                comparison["accuracy_comparison"] = {
                    "pytorch_map_50": pytorch_map,
                    "onnx_map_50": onnx_map,
                    "map_diff": map_diff,
                    "map_diff_pct": map_diff / pytorch_map * 100
                    if pytorch_map > 0
                    else 0,
                }

                if map_diff < 0.01:
                    comparison["recommendations"].append(
                        "✅ ONNX 精度损失 < 1%，可放心使用"
                    )
                elif map_diff < 0.03:
                    comparison["recommendations"].append(
                        f"⚖️ ONNX 精度损失 {map_diff * 100:.1f}%，在可接受范围内"
                    )
                else:
                    comparison["recommendations"].append(
                        f"⚠️ ONNX 精度损失较大 ({map_diff * 100:.1f}%)，请检查导出参数"
                    )

        return comparison

    def _print_summary(self, comparison: Dict[str, Any]):
        """打印对比摘要"""
        if not self.logger:
            return

        self.logger.info("\n" + "=" * 70)
        self.logger.info("格式对比结果摘要")
        self.logger.info("=" * 70)

        # 性能表格
        self.logger.info("\n性能对比:")
        self.logger.info("-" * 70)
        self.logger.info(f"{'格式':<15} {'FPS':<12} {'延迟(ms)':<12} {'mAP@0.5':<12}")
        self.logger.info("-" * 70)

        for fmt, metrics in comparison["formats"].items():
            self.logger.info(
                f"{fmt.upper():<15} "
                f"{metrics['fps']:<12.2f} "
                f"{metrics['avg_time_ms']:<12.2f} "
                f"{metrics['map_50']:<12.4f}"
            )

        # 速度提升
        speed = comparison.get("speed_comparison", {})
        if speed:
            self.logger.info("\n速度提升:")
            self.logger.info(
                f"  ONNX vs PyTorch: {speed['speedup_pct']:+.1f}% "
                f"({speed['speedup']:.2f}x)"
            )

        # 推荐
        if comparison["recommendations"]:
            self.logger.info("\n推荐:")
            for rec in comparison["recommendations"]:
                self.logger.info(f"  {rec}")

        self.logger.info("=" * 70)

    def save_report(self, output_file: str = None) -> str:
        """保存对比报告"""
        if output_file is None:
            output_file = self.output_dir / f"{self.model_name}_format_comparison.json"

        report = {
            "model_name": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "results": self.results,
            "comparison": self._generate_comparison(),
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return str(output_file)


def compare_model_formats_cli(
    model_path: str,
    model_name: str = None,
    formats: List[str] = None,
    config: str = "config.yaml",
    num_images: int = 50,
    output_dir: str = "outputs/format_comparison",
) -> Dict[str, Any]:
    """
    CLI 格式对比接口

    Args:
        model_path: 模型文件路径 (.pt)
        model_name: 模型名称（默认为文件名）
        formats: 要对比的格式
        config: 配置文件路径
        num_images: 测试图片数
        output_dir: 输出目录

    Returns:
        对比结果
    """
    from src.utils.logger import Config, setup_logger

    # 设置日志
    try:
        config_obj = Config(config)
        logger = setup_logger(config_obj)
    except:
        import logging

        logger = logging.getLogger("format_comparison")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    model_path = Path(model_path)
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        return {"success": False, "error": "Model not found"}

    if model_name is None:
        model_name = model_path.stem

    # 加载数据集
    try:
        config_obj = Config(config)
        dataset_config = config_obj.get_dataset_config()
        dataset = COCOInferenceDataset(dataset_config["path"], dataset_config["split"])

        annotations_file = (
            Path(dataset_config["path"]).expanduser()
            / "annotations"
            / f"instances_{dataset_config['split']}.json"
        )
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        return {"success": False, "error": str(e)}

    # 执行对比
    comparer = FormatComparison(model_name, str(model_path), output_dir, logger)
    results = comparer.compare_formats(
        dataset=dataset,
        annotations_file=str(annotations_file),
        formats=formats,
        max_images=num_images,
    )

    # 保存报告
    report_file = comparer.save_report()
    logger.info(f"\n报告已保存: {report_file}")

    return results
