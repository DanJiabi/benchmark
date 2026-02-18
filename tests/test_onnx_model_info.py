#!/usr/bin/env python3
"""测试 ONNX 模型信息获取"""

import unittest
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models import create_model


class TestONNXModelInfo(unittest.TestCase):
    """测试 ONNX 模型的 model_info"""

    def setUp(self):
        """测试设置"""
        self.onnx_model_path = Path("models_export/yolov8n.onnx")
        if not self.onnx_model_path.exists():
            self.skipTest(f"ONNX 模型不存在: {self.onnx_model_path}")

    def test_model_info_has_model_size_mb(self):
        """测试 model_info 包含 model_size_mb"""
        model = create_model(str(self.onnx_model_path), device="cpu")
        model_info = model.get_model_info()

        self.assertIn("model_size_mb", model_info)
        self.assertGreater(model_info["model_size_mb"], 0)
        self.assertIsInstance(model_info["model_size_mb"], (int, float))

    def test_model_info_has_params_estimate(self):
        """测试 model_info 包含 params 估算值"""
        model = create_model(str(self.onnx_model_path), device="cpu")
        model_info = model.get_model_info()

        self.assertIn("params", model_info)
        self.assertGreater(model_info["params"], 0)
        self.assertIsInstance(model_info["params"], (int, float))

    def test_params_is_reasonable_estimate(self):
        """测试 params 估算值在合理范围内"""
        model = create_model(str(self.onnx_model_path), device="cpu")
        model_info = model.get_model_info()

        # YOLOv8n 的参数量应该在 3-4M 左右
        # 文件大小约 6MB，估算的 params 应该在 6 左右（粗略估算）
        self.assertGreater(model_info["params"], 0)
        self.assertLess(model_info["params"], 1000)  # 不应该超过 1000M

    def test_model_size_mb_matches_file_size(self):
        """测试 model_size_mb 与实际文件大小一致"""
        model = create_model(str(self.onnx_model_path), device="cpu")
        model_info = model.get_model_info()

        # 计算实际文件大小
        actual_size_bytes = self.onnx_model_path.stat().st_size
        actual_size_mb = actual_size_bytes / (1024 * 1024)

        # 允许 ±0.01 MB 的误差（四舍五入差异）
        self.assertAlmostEqual(
            model_info["model_size_mb"],
            round(actual_size_mb, 2),
            places=2,
            msg="model_size_mb 应该与实际文件大小一致",
        )

    def test_model_info_has_required_fields(self):
        """测试 model_info 包含所有必需字段"""
        model = create_model(str(self.onnx_model_path), device="cpu")
        model_info = model.get_model_info()

        required_fields = ["name", "weights", "format", "params", "model_size_mb"]
        for field in required_fields:
            self.assertIn(field, model_info, f"缺少必需字段: {field}")


if __name__ == "__main__":
    unittest.main()
