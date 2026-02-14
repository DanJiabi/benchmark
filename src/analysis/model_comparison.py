"""
模型对比分析模块

提供基准模型与用户模型之间的性能对比分析功能。
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

import numpy as np

from src.models.base import BaseModel
from src.models import create_model, UserModelLoader
from src.data.coco_dataset import COCOInferenceDataset
from src.metrics.coco_metrics import COCOMetrics, PerformanceMetrics


class ModelComparison:
    """模型对比分析类"""

    def __init__(
        self,
        baseline_model: BaseModel,
        user_model: BaseModel,
        logger,
    ):
        self.baseline_model = baseline_model
        self.user_model = user_model
        self.logger = logger

        self.baseline_results = None
        self.user_results = None
        self.comparison = None

    def run_comparison(
        self,
        dataset: COCOInferenceDataset,
        annotations_file: str,
        max_images: Optional[int] = None,
        conf_threshold: float = 0.001,
    ) -> None:
        """
        运行模型对比

        Args:
            dataset: 数据集
            annotations_file: COCO 标注文件路径
            max_images: 最大图片数量
            conf_threshold: 置信度阈值
        """
        self.logger.info("=" * 70)
        self.logger.info("开始模型对比分析")
        self.logger.info("=" * 70)

        # 运行基准模型
        self.logger.info(
            f"评估基准模型: {self.baseline_model.model_info.get('name', 'Unknown')}"
        )
        self.baseline_results = self._evaluate_model(
            self.baseline_model, dataset, annotations_file, max_images, conf_threshold
        )

        if self.baseline_results is None:
            self.logger.error("基准模型评估失败")
            return

        # 运行用户模型
        self.logger.info(
            f"评估用户模型: {self.user_model.model_info.get('name', 'Unknown')}"
        )
        self.user_results = self._evaluate_model(
            self.user_model, dataset, annotations_file, max_images, conf_threshold
        )

        if self.user_results is None:
            self.logger.error("用户模型评估失败")
            return

        # 生成对比分析
        self.logger.info("生成对比分析...")
        self.comparison = self._generate_comparison()

        # 输出对比结果
        self._log_comparison_results()

    def _evaluate_model(
        self,
        model: BaseModel,
        dataset: COCOInferenceDataset,
        annotations_file: str,
        max_images: Optional[int],
        conf_threshold: float,
    ) -> Optional[Dict[str, Any]]:
        """评估单个模型"""
        try:
            coco_metrics_calculator = COCOMetrics(annotations_file)
            perf_metrics = PerformanceMetrics()

            total_images = max_images if max_images else len(dataset)
            self.logger.info(f"将处理 {total_images} 张图片")

            all_detections = {}

            for idx, (image_id, image) in enumerate(dataset):
                if idx >= total_images:
                    break

                try:
                    detections = model.predict(image, conf_threshold)

                    if detections:
                        all_detections[image_id] = detections

                except Exception as e:
                    self.logger.warning(f"  图片 {idx} 推理失败: {e}")
                    continue

            # 计算 COCO 指标
            predictions = coco_metrics_calculator.predictions_to_coco_format(
                all_detections
            )
            coco_metrics = coco_metrics_calculator.compute_metrics(predictions)

            # 计算性能指标
            performance_stats = perf_metrics.compute_performance_stats()

            model_info = model.get_model_info()

            return {
                "model_name": model_info.get("name", "Unknown"),
                "coco_metrics": coco_metrics,
                "performance": performance_stats,
                "model_info": model_info,
                "num_images": total_images,
                "num_detections": len(all_detections),
            }

        except Exception as e:
            self.logger.error(f"模型评估失败: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _generate_comparison(self) -> Dict[str, Any]:
        """生成对比分析结果"""
        baseline = self.baseline_results
        user = self.user_results

        # 基础指标对比
        comparison = {
            "baseline_name": baseline["model_name"],
            "user_model_name": user["model_name"],
            "timestamp": datetime.now().isoformat(),
            "metrics_comparison": {},
            "performance_comparison": {},
            "recommendations": [],
        }

        # mAP 对比
        for metric in ["AP@0.50", "AP@0.50:0.95", "AP@0.50:0.05", "AP@0.75"]:
            baseline_val = baseline["coco_metrics"].get(metric, 0)
            user_val = user["coco_metrics"].get(metric, 0)

            diff = user_val - baseline_val
            diff_pct = (diff / baseline_val * 100) if baseline_val > 0 else 0

            comparison["metrics_comparison"][metric] = {
                "baseline": baseline_val,
                "user": user_val,
                "diff": diff,
                "diff_pct": diff_pct,
            }

        # 性能对比
        baseline_fps = baseline["performance"].get("fps", 0)
        user_fps = user["performance"].get("fps", 0)

        comparison["performance_comparison"] = {
            "baseline_fps": baseline_fps,
            "user_fps": user_fps,
            "fps_diff": user_fps - baseline_fps,
            "fps_diff_pct": ((user_fps - baseline_fps) / baseline_fps * 100)
            if baseline_fps > 0
            else 0,
            "speedup": user_fps / baseline_fps if baseline_fps > 0 else 0,
            "baseline_avg_time": baseline["performance"].get("avg_time_ms", 0),
            "user_avg_time": user["performance"].get("avg_time_ms", 0),
        }

        # 生成推荐
        comparison["recommendations"] = self._generate_recommendations(comparison)

        return comparison

    def _generate_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """生成使用推荐"""
        recommendations = []

        # 准确性分析
        map_50_diff = comparison["metrics_comparison"]["AP@0.50"]["diff"]
        map_diff = comparison["metrics_comparison"]["AP@0.50:0.95"]["diff"]

        if map_diff > 0.05:  # 用户模型更好
            recommendations.append("✅ 用户模型在 mAP@0.50 上有明显提升")
        elif map_diff > 0:
            recommendations.append("✅ 用户模型在 mAP@0.50 上有所提升")
        elif map_diff < -0.05:  # 基准模型更好
            recommendations.append("⚠️  用户模型在 mAP@0.50 上明显低于基准")
        elif map_diff < -0.01:
            recommendations.append("⚠️  用户模型在 mAP@0.50 上略低于基准")

        # 速度分析
        fps_diff_pct = comparison["performance_comparison"]["fps_diff_pct"]
        speedup = comparison["performance_comparison"]["speedup"]

        if fps_diff_pct > 50:  # 用户模型快很多
            recommendations.append(f"✅ 用户模型速度快 {fps_diff_pct:.1f}%")
            recommendations.append(f"✅ 用户模型速度是基准的 {speedup:.2f}x")
        elif fps_diff_pct > 10:  # 用户模型明显更快
            recommendations.append(f"✅ 用户模型速度快 {fps_diff_pct:.1f}%")
        elif fps_diff_pct < -50:  # 用户模型慢很多
            recommendations.append(f"⚠️  用户模型速度慢 {abs(fps_diff_pct):.1f}%")
            recommendations.append(f"⚠️  用户模型速度是基准的 {1 / speedup:.2f}x")
        elif fps_diff_pct < -10:  # 用户模型明显更慢
            recommendations.append(f"⚠️  用户模型速度慢 {abs(fps_diff_pct):.1f}%")

        # 模型大小分析
        baseline_params = self.baseline_results["model_info"].get("params", 0)
        user_params = self.user_results["model_info"].get("params", 0)

        if baseline_params > 0:
            param_ratio = user_params / baseline_params
            if param_ratio < 0.8:  # 用户模型明显更小
                recommendations.append(f"✅ 用户模型参数量更少 ({param_ratio:.2f}x)")
            elif param_ratio > 1.2:  # 用户模型明显更大
                recommendations.append(f"⚠️  用户模型参数量更多 ({param_ratio:.2f}x)")

        # 综合推荐
        if map_diff > 0 and fps_diff_pct > 0:
            recommendations.append("🎉 用户模型在准确率和速度上都优于基准")
        elif map_diff > 0:
            recommendations.append("✅ 用户模型准确率更高，建议采用")
        elif map_diff < 0 and fps_diff_pct > 0:
            recommendations.append("⚖️  权衡：用户模型更快但准确率略低")
        elif map_diff < 0:
            recommendations.append("❌ 用户模型准确率低于基准，需要改进")

        return recommendations

    def _log_comparison_results(self) -> None:
        """输出对比结果"""
        if not self.comparison:
            return

        comp = self.comparison

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("模型对比结果")
        self.logger.info("=" * 70)

        # 基本信息
        self.logger.info(f"基准模型: {comp['baseline_name']}")
        self.logger.info(f"用户模型: {comp['user_model_name']}")
        self.logger.info(f"测试时间: {comp['timestamp']}")

        # mAP 对比
        self.logger.info("")
        self.logger.info("mAP 指标对比:")
        self.logger.info("-" * 70)

        for metric in ["AP@0.50", "AP@0.50:0.95"]:
            baseline = comp["metrics_comparison"][metric]["baseline"]
            user = comp["metrics_comparison"][metric]["user"]
            diff = comp["metrics_comparison"][metric]["diff"]
            diff_pct = comp["metrics_comparison"][metric]["diff_pct"]

            baseline_str = f"{baseline:.4f}"
            user_str = f"{user:.4f}"
            diff_str = f"{diff:+.4f} ({diff_pct:+.2f}%)"

            self.logger.info(
                f"  {metric:20s} | 基准: {baseline_str} | 用户: {user_str} | 差异: {diff_str}"
            )

        # 性能对比
        self.logger.info("")
        self.logger.info("性能指标对比:")
        self.logger.info("-" * 70)

        perf = comp["performance_comparison"]
        self.logger.info(f"  基准 FPS:     {perf['baseline_fps']:.2f}")
        self.logger.info(f"  用户 FPS:     {perf['user_fps']:.2f}")
        self.logger.info(f"  FPS 差异:     {perf['fps_diff']:+.2f}")
        self.logger.info(f"  FPS 提升:      {perf['fps_diff_pct']:+.2f}%")
        self.logger.info(f"  加速比:       {perf['speedup']:.2f}x")
        self.logger.info(f"  基准平均时间: {perf['baseline_avg_time']:.2f}ms")
        self.logger.info(f"  用户平均时间: {perf['user_avg_time']:.2f}ms")

        # 推荐
        self.logger.info("")
        self.logger.info("推荐:")
        self.logger.info("-" * 70)

        for rec in comp["recommendations"]:
            self.logger.info(f"  {rec}")

        self.logger.info("")
        self.logger.info("=" * 70)

    def save_results(
        self, output_path: Path, formats: List[str] = ["json", "html"]
    ) -> None:
        """
        保存对比结果

        Args:
            output_path: 输出目录
            formats: 输出格式列表（json, html, csv）
        """
        if not self.comparison:
            self.logger.warning("没有对比结果可保存")
            return

        output_path.mkdir(parents=True, exist_ok=True)

        # JSON
        if "json" in formats:
            json_file = output_path / "comparison.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(self.comparison, f, indent=2, ensure_ascii=False)
            self.logger.info(f"JSON 结果已保存: {json_file}")

        # HTML
        if "html" in formats:
            html_file = output_path / "comparison.html"
            html_content = self._generate_html_report()
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            self.logger.info(f"HTML 报告已保存: {html_file}")

        # CSV
        if "csv" in formats:
            csv_file = output_path / "comparison.csv"
            csv_content = self._generate_csv_report()
            with open(csv_file, "w", encoding="utf-8") as f:
                f.write(csv_content)
            self.logger.info(f"CSV 报告已保存: {csv_file}")

    def _generate_html_report(self) -> str:
        """生成 HTML 报告"""
        if not self.comparison:
            return ""

        comp = self.comparison
        baseline = self.baseline_results
        user = self.user_results

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
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #34495e;
            padding-bottom: 8px;
            margin-top: 30px;
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
        .metric-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .metric-table th {{
            background: #343a40;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        .metric-table td {{
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }}
        .positive {{ color: #28a745; }}
        .negative {{ color: #dc3545; }}
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
        .timestamp {{
            color: #6c757d;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 模型对比分析报告</h1>
        <p class="timestamp">生成时间: {comp["timestamp"]}</p>

        <div class="info-grid">
            <div class="info-card">
                <h3>基准模型</h3>
                <p><strong>名称:</strong> {comp["baseline_name"]}</p>
                <p><strong>mAP@0.50:</strong> {baseline["coco_metrics"]["AP@0.50"]:.4f}</p>
                <p><strong>mAP@0.50:0.95:</strong> {baseline["coco_metrics"]["AP@0.50:0.95"]:.4f}</p>
                <p><strong>FPS:</strong> {baseline["performance"]["fps"]:.2f}</p>
                <p><strong>参数量:</strong> {baseline["model_info"].get("params", 0):.2f}M</p>
            </div>
            <div class="info-card">
                <h3>用户模型</h3>
                <p><strong>名称:</strong> {comp["user_model_name"]}</p>
                <p><strong>mAP@0.50:</strong> {user["coco_metrics"]["AP@0.50"]:.4f}</p>
                <p><strong>mAP@0.50:0.95:</strong> {user["coco_metrics"]["AP@0.50:0.95"]:.4f}</p>
                <p><strong>FPS:</strong> {user["performance"]["fps"]:.2f}</p>
                <p><strong>参数量:</strong> {user["model_info"].get("params", 0):.2f}M</p>
            </div>
        </div>

        <h2>📈 指标对比</h2>
        <table class="metric-table">
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
                    <td>{comp["metrics_comparison"]["AP@0.50"]["baseline"]:.4f}</td>
                    <td>{comp["metrics_comparison"]["AP@0.50"]["user"]:.4f}</td>
                    <td class="{"positive" if comp["metrics_comparison"]["AP@0.50"]["diff"] > 0 else "negative"}">{comp["metrics_comparison"]["AP@0.50"]["diff"]:+.4f}</td>
                    <td class="{"positive" if comp["metrics_comparison"]["AP@0.50"]["diff_pct"] > 0 else "negative"}">{comp["metrics_comparison"]["AP@0.50"]["diff_pct"]:+.2f}%</td>
                </tr>
                <tr>
                    <td><strong>mAP@0.50:0.95</strong></td>
                    <td>{comp["metrics_comparison"]["AP@0.50:0.95"]["baseline"]:.4f}</td>
                    <td>{comp["metrics_comparison"]["AP@0.50:0.95"]["user"]:.4f}</td>
                    <td class="{"positive" if comp["metrics_comparison"]["AP@0.50:0.95"]["diff"] > 0 else "negative"}">{comp["metrics_comparison"]["AP@0.50:0.95"]["diff"]:+.4f}</td>
                    <td class="{"positive" if comp["metrics_comparison"]["AP@0.50:0.95"]["diff_pct"] > 0 else "negative"}">{comp["metrics_comparison"]["AP@0.50:0.95"]["diff_pct"]:+.2f}%</td>
                </tr>
                <tr>
                    <td><strong>FPS</strong></td>
                    <td>{comp["performance_comparison"]["baseline_fps"]:.2f}</td>
                    <td>{comp["performance_comparison"]["user_fps"]:.2f}</td>
                    <td class="{"positive" if comp["performance_comparison"]["fps_diff"] > 0 else "negative"}">{comp["performance_comparison"]["fps_diff"]:+.2f}</td>
                    <td class="{"positive" if comp["performance_comparison"]["fps_diff_pct"] > 0 else "negative"}">{comp["performance_comparison"]["fps_diff_pct"]:+.2f}%</td>
                </tr>
                <tr>
                    <td><strong>加速比</strong></td>
                    <td>1.00x</td>
                    <td>{comp["performance_comparison"]["speedup"]:.2f}x</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
            </tbody>
        </table>

        <h2>💡 推荐与建议</h2>
        <div class="recommendations">
"""

        for rec in comp["recommendations"]:
            html += f'            <div class="recommendation-item">{rec}</div>\n'

        html += f"""        </div>

        <p style="text-align: center; color: #6c757d; font-style: italic; margin-top: 40px;">
            报告由 od-benchmark 生成
        </p>
    </div>
</body>
</html>
"""

        return html

    def _generate_csv_report(self) -> str:
        """生成 CSV 报告"""
        if not self.comparison:
            return ""

        comp = self.comparison

        lines = []
        lines.append("指标,基准模型,用户模型,差异,变化%")
        lines.append("-" * 60)

        for metric in ["AP@0.50", "AP@0.50:0.95"]:
            baseline = comp["metrics_comparison"][metric]["baseline"]
            user = comp["metrics_comparison"][metric]["user"]
            diff = comp["metrics_comparison"][metric]["diff"]
            diff_pct = comp["metrics_comparison"][metric]["diff_pct"]
            lines.append(
                f"{metric},{baseline:.4f},{user:.4f},{diff:+.4f},{diff_pct:+.2f}%"
            )

        lines.append("")
        lines.append("性能指标")
        lines.append(f"基准 FPS,{comp['performance_comparison']['baseline_fps']:.2f}")
        lines.append(f"用户 FPS,{comp['performance_comparison']['user_fps']:.2f}")
        lines.append(f"FPS 差异,{comp['performance_comparison']['fps_diff']:+.2f}")
        lines.append(f"FPS 提升,{comp['performance_comparison']['fps_diff_pct']:+.2f}%")
        lines.append(f"加速比,{comp['performance_comparison']['speedup']:.2f}x")

        lines.append("")
        lines.append("推荐")
        for rec in comp["recommendations"]:
            lines.append(rec)

        return "\n".join(lines)

    def get_comparison(self) -> Dict[str, Any]:
        """获取对比结果"""
        return self.comparison if self.comparison else {}
