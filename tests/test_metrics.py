"""
指标计算单元测试
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.models.base import Detection
from src.metrics.coco_metrics import COCOMetrics, PerformanceMetrics


class TestDetection:
    """Detection 类测试"""

    def test_detection_creation(self):
        """测试 Detection 对象创建"""
        bbox = [100, 100, 200, 200]
        confidence = 0.95
        class_id = 0

        det = Detection(bbox, confidence, class_id)

        assert det.bbox == bbox
        assert det.confidence == confidence
        assert det.class_id == class_id

    def test_detection_types(self):
        """测试 Detection 数据类型"""
        det = Detection([100, 100, 200, 200], 0.95, 0)

        assert isinstance(det.bbox, list)
        assert len(det.bbox) == 4
        assert isinstance(det.confidence, float)
        assert isinstance(det.class_id, int)


class TestPerformanceMetrics:
    """PerformanceMetrics 测试"""

    @pytest.fixture
    def perf_metrics(self):
        """创建 PerformanceMetrics 对象"""
        return PerformanceMetrics()

    @pytest.fixture
    def sample_times(self):
        """生成示例推理时间"""
        return [0.01, 0.015, 0.02, 0.018, 0.025]

    def test_metrics_initialization(self, perf_metrics):
        """测试指标初始化"""
        assert perf_metrics is not None
        assert perf_metrics.inference_times == []
        assert perf_metrics.total_images == 0

    def test_timer_operations(self, perf_metrics):
        """测试计时器操作"""
        import time

        start = perf_metrics.start_timer()
        time.sleep(0.001)
        elapsed = perf_metrics.end_timer(start)

        assert elapsed > 0
        assert elapsed < 1

    def test_add_inference_time(self, perf_metrics, sample_times):
        """测试添加推理时间"""
        for t in sample_times:
            perf_metrics.add_inference_time(t)

        assert len(perf_metrics.inference_times) == len(sample_times)
        assert perf_metrics.total_images == len(sample_times)

    def test_compute_performance_stats(self, perf_metrics, sample_times):
        """测试计算性能统计"""
        for t in sample_times:
            perf_metrics.add_inference_time(t)

        stats = perf_metrics.compute_performance_stats()

        assert isinstance(stats, dict)
        assert "avg_inference_time_ms" in stats
        assert "min_inference_time_ms" in stats
        assert "max_inference_time_ms" in stats
        assert "std_inference_time_ms" in stats
        assert "fps" in stats
        assert "total_images" in stats

    def test_fps_calculation(self, perf_metrics):
        """测试 FPS 计算"""
        perf_metrics.add_inference_time(0.01)
        perf_metrics.add_inference_time(0.02)

        stats = perf_metrics.compute_performance_stats()

        assert stats["fps"] > 0
        assert stats["fps"] < 100

    def test_empty_metrics(self, perf_metrics):
        """测试空指标的处理"""
        stats = perf_metrics.compute_performance_stats()

        assert stats == {}

    def test_stats_values(self, perf_metrics, sample_times):
        """测试统计值的正确性"""
        for t in sample_times:
            perf_metrics.add_inference_time(t)

        stats = perf_metrics.compute_performance_stats()

        avg_ms = stats["avg_inference_time_ms"]
        min_ms = stats["min_inference_time_ms"]
        max_ms = stats["max_inference_time_ms"]
        std_ms = stats["std_inference_time_ms"]

        assert min_ms <= avg_ms <= max_ms
        assert std_ms >= 0
        assert stats["total_images"] == len(sample_times)


class TestCOCOMetrics:
    """COCO 指标测试"""

    @pytest.fixture
    def coco_metrics(self):
        """创建 COCO 指标对象"""
        annotations_file = "/Users/danjiabi/raw/COCO/annotations/instances_val2017.json"
        return COCOMetrics(annotations_file)

    def test_coco_initialization(self, coco_metrics):
        """测试 COCO 初始化"""
        assert coco_metrics is not None
        assert coco_metrics.coco_gt is not None
        assert coco_metrics.image_ids is not None

    def test_predictions_to_coco_format(self, coco_metrics):
        """测试 COCO 格式转换"""
        detections_dict = {1: [Detection([100, 100, 200, 200], 0.95, 0)]}

        predictions = coco_metrics.predictions_to_coco_format(detections_dict)

        assert isinstance(predictions, list)
        assert len(predictions) > 0

        pred = predictions[0]
        assert "image_id" in pred
        assert "category_id" in pred
        assert "bbox" in pred
        assert "score" in pred
        assert "id" in pred

    def test_bbox_format_conversion(self, coco_metrics):
        """测试边界框格式转换"""
        detections_dict = {1: [Detection([100, 100, 200, 200], 0.95, 0)]}

        predictions = coco_metrics.predictions_to_coco_format(detections_dict)
        bbox = predictions[0]["bbox"]

        assert isinstance(bbox, list)
        assert len(bbox) == 4
        assert bbox[0] == 100  # x1
        assert bbox[1] == 100  # y1
        assert bbox[2] == 100  # width (x2 - x1)
        assert bbox[3] == 100  # height (y2 - y1)

    def test_category_id_offset(self, coco_metrics):
        """测试类别 ID 偏移（COCO 从1开始）"""
        detections_dict = {1: [Detection([100, 100, 200, 200], 0.95, 1)]}

        predictions = coco_metrics.predictions_to_coco_format(detections_dict)
        category_id = predictions[0]["category_id"]

        # COCO category_id 从 1 开始，模型应该直接输出正确的 ID
        assert category_id == 1

    def test_score_range(self, coco_metrics):
        """测试分数范围"""
        detections_dict = {
            1: [
                Detection([100, 100, 200, 200], 0.5, 1),
                Detection([100, 100, 200, 200], 1.0, 2),
            ]
        }

        predictions = coco_metrics.predictions_to_coco_format(detections_dict)

        for pred in predictions:
            score = pred["score"]
            assert 0 <= score <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
