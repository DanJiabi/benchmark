"""
可视化工具单元测试
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.utils.visualization import (
    draw_detection_boxes,
    save_detection_visualization,
    generate_results_table,
)
from src.models.base import Detection


class TestDrawDetectionBoxes:
    """绘制检测框测试"""

    @pytest.fixture
    def sample_image(self):
        """生成测试图片"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_detections(self):
        """生成测试检测框"""
        return [
            Detection([100, 100, 200, 200], 0.95, 0),
            Detection([300, 150, 400, 250], 0.87, 1),
            Detection([50, 300, 150, 400], 0.72, 2),
        ]

    @pytest.fixture
    def class_names(self):
        """类别名称映射"""
        return {0: "person", 1: "car", 2: "dog"}

    def test_draw_boxes(self, sample_image, sample_detections, class_names):
        """测试绘制检测框"""
        result = draw_detection_boxes(sample_image, sample_detections, class_names)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image.shape

    def test_draw_boxes_no_class_names(self, sample_image, sample_detections):
        """测试无类别名称绘制"""
        result = draw_detection_boxes(sample_image, sample_detections, {})

        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_draw_boxes_empty_detections(self, sample_image, class_names):
        """测试空检测框列表"""
        result = draw_detection_boxes(sample_image, [], class_names)

        assert result is not None
        # 应该返回原图
        assert np.array_equal(result, sample_image)

    def test_draw_boxes_max_boxes(self, sample_image, sample_detections, class_names):
        """测试最大检测框数量限制"""
        # 生成大量检测框
        many_detections = [
            Detection([i * 10, i * 10, i * 10 + 50, i * 10 + 50], 0.5, 0)
            for i in range(20)
        ]

        result = draw_detection_boxes(
            sample_image, many_detections, class_names, max_boxes=5
        )
        assert result is not None


class TestSaveDetectionVisualization:
    """保存可视化测试"""

    @pytest.fixture
    def sample_image(self):
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_detections(self):
        return [Detection([100, 100, 200, 200], 0.95, 0)]

    def test_save_visualization(self, sample_image, sample_detections, tmp_path):
        """测试保存可视化"""
        output_path = tmp_path / "test_viz.jpg"
        class_names = {0: "person"}

        count = save_detection_visualization(
            sample_image, sample_detections, class_names, output_path
        )

        assert count == 1
        assert output_path.exists()


class TestGenerateResultsTable:
    """生成结果表格测试"""

    @pytest.fixture
    def sample_results(self):
        """生成测试结果数据"""
        return {
            "yolov8n": {
                "coco_metrics": {"AP@0.50": 0.5234, "AP@0.50:0.95": 0.3689},
                "performance": {"fps": 125.5},
                "model_info": {"params": 3.2},
            },
            "yolov8s": {
                "coco_metrics": {"AP@0.50": 0.6123, "AP@0.50:0.95": 0.4456},
                "performance": {"fps": 95.3},
                "model_info": {"params": 11.2},
            },
        }

    def test_generate_table(self, sample_results):
        """测试生成表格"""
        df = generate_results_table(sample_results)

        assert df is not None
        assert len(df) == 2
        assert "yolov8n" in df.index
        assert "yolov8s" in df.index

    def test_table_columns(self, sample_results):
        """测试表格列"""
        df = generate_results_table(sample_results)

        assert "AP@0.50" in df.columns
        assert "fps" in df.columns
        assert "params" in df.columns

    def test_empty_results(self):
        """测试空结果"""
        df = generate_results_table({})

        assert df is not None
        assert len(df) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
