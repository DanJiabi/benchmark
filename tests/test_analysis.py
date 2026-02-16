"""
分析功能单元测试
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import logging

from src.analysis.model_comparison import ModelComparison
from src.models.base import BaseModel, Detection


class MockModel(BaseModel):
    """模拟模型用于测试"""

    def load_model(self, weights_path):
        pass

    def predict(self, image, conf_threshold=None):
        return []

    def get_model_info(self):
        return {"name": "mock_model"}

    def warmup(self, image_size=(640, 640)):
        pass


class TestModelComparison:
    """模型对比测试"""

    @pytest.fixture
    def mock_models(self):
        """创建模拟模型"""
        baseline = MockModel(device="cpu")
        user = MockModel(device="cpu")
        logger = logging.getLogger("test")
        return baseline, user, logger

    @pytest.fixture
    def sample_results(self):
        """样本测试结果"""
        return {
            "model1": {
                "coco_metrics": {
                    "AP@0.50": 0.5234,
                    "AP@0.50:0.95": 0.3689,
                    "AP@small": 0.1234,
                    "AP@medium": 0.3456,
                    "AP@large": 0.5678,
                },
                "performance": {
                    "fps": 125.5,
                    "avg_inference_time_ms": 7.97,
                },
                "model_info": {
                    "params": 3.2,
                    "model_size_mb": 6.2,
                },
            },
            "model2": {
                "coco_metrics": {
                    "AP@0.50": 0.6123,
                    "AP@0.50:0.95": 0.4456,
                    "AP@small": 0.2345,
                    "AP@medium": 0.4567,
                    "AP@large": 0.6789,
                },
                "performance": {
                    "fps": 95.3,
                    "avg_inference_time_ms": 10.49,
                },
                "model_info": {
                    "params": 11.2,
                    "model_size_mb": 22.5,
                },
            },
        }

    def test_comparison_initialization(self, mock_models):
        """测试对比初始化"""
        baseline, user, logger = mock_models
        comparison = ModelComparison(baseline, user, logger)
        assert comparison is not None
        assert comparison.baseline_model == baseline
        assert comparison.user_model == user

    def test_compare_metrics(self, mock_models, sample_results):
        """测试指标对比"""
        # 对比两个模型
        baseline = sample_results["model1"]
        user_model = sample_results["model2"]

        # 计算差异
        map_diff = (
            user_model["coco_metrics"]["AP@0.50:0.95"]
            - baseline["coco_metrics"]["AP@0.50:0.95"]
        )
        fps_diff = user_model["performance"]["fps"] - baseline["performance"]["fps"]

        assert map_diff > 0  # model2 应该更好
        assert fps_diff < 0  # model2 应该更慢

    def test_speedup_calculation(self, sample_results):
        """测试加速比计算"""
        baseline_fps = sample_results["model1"]["performance"]["fps"]
        user_fps = sample_results["model2"]["performance"]["fps"]

        speedup = user_fps / baseline_fps

        assert speedup > 0
        assert speedup < 1  # model2 应该更慢

    def test_accuracy_improvement(self, sample_results):
        """测试精度提升"""
        baseline_map = sample_results["model1"]["coco_metrics"]["AP@0.50:0.95"]
        user_map = sample_results["model2"]["coco_metrics"]["AP@0.50:0.95"]

        improvement = (user_map - baseline_map) / baseline_map * 100

        assert improvement > 0  # model2 精度更高


class TestRecommendations:
    """推荐逻辑测试"""

    def test_better_accuracy_recommendation(self):
        """测试更高精度推荐"""
        baseline_map = 0.3689
        user_map = 0.4456

        diff_pct = (user_map - baseline_map) / baseline_map * 100

        assert diff_pct > 5  # 提升超过 5%
        # 应该推荐使用用户模型

    def test_faster_speed_recommendation(self):
        """测试更快速度推荐"""
        baseline_fps = 95.3
        user_fps = 125.5

        speedup_pct = (user_fps - baseline_fps) / baseline_fps * 100

        assert speedup_pct > 10  # 速度提升超过 10%
        # 应该推荐使用用户模型

    def test_balanced_recommendation(self):
        """测试平衡推荐"""
        # 精度更高但速度更慢
        baseline = {"map": 0.3689, "fps": 125.5}
        user = {"map": 0.4456, "fps": 95.3}

        # 需要根据应用场景权衡
        assert user["map"] > baseline["map"]
        assert user["fps"] < baseline["fps"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
