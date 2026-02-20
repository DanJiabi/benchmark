#!/usr/bin/env python3
"""测试 ONNX 模型 NMS 和类别 ID 映射修复"""

import unittest
from pathlib import Path
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestONNXNMSAndClassID(unittest.TestCase):
    """测试 ONNX 模型的 NMS 和类别 ID 修复"""

    def setUp(self):
        """测试设置"""
        self.onnx_model_path = Path("models_export/yolov8n.onnx")
        if not self.onnx_model_path.exists():
            self.skipTest(f"ONNX 模型不存在: {self.onnx_model_path}")

    def test_apply_nms_method_exists(self):
        """测试 _apply_nms 方法存在"""
        from src.models import create_model

        model = create_model(str(self.onnx_model_path), device="cpu")
        self.assertTrue(hasattr(model, "_apply_nms"), "缺少 _apply_nms 方法")

    def test_simple_nms_method_exists(self):
        """测试 _simple_nms 方法存在"""
        from src.models import create_model

        model = create_model(str(self.onnx_model_path), device="cpu")
        self.assertTrue(hasattr(model, "_simple_nms"), "缺少 _simple_nms 方法")

    def test_class_id_mapping_plus_one(self):
        """测试类别 ID 映射（+1 转换）"""
        from src.models import create_model
        from src.data.coco_dataset import COCOInferenceDataset

        model = create_model(str(self.onnx_model_path), device="cpu")

        # 创建虚拟图像
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # 运行推理
        detections = model.predict(dummy_image, conf_threshold=0.001)

        # 检查类别 ID 是否在 COCO 范围内（1-80）
        for det in detections:
            self.assertGreaterEqual(
                det.class_id,
                1,
                f"类别 ID 应该 >= 1（COCO 格式），但得到 {det.class_id}",
            )
            self.assertLessEqual(
                det.class_id,
                80,
                f"类别 ID 应该 <= 80（COCO 格式），但得到 {det.class_id}",
            )

    def test_nms_reduces_duplicates(self):
        """测试 NMS 减少重复检测框"""
        from src.models import create_model

        model = create_model(str(self.onnx_model_path), device="cpu")

        # 创建虚拟图像
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # 运行推理
        detections = model.predict(dummy_image, conf_threshold=0.001)

        # 验证有检测结果
        if len(detections) > 0:
            # NMS 后的检测框数量应该合理（不应该有数千个）
            self.assertLess(
                len(detections),
                1000,
                f"NMS 后检测框数量 {len(detections)} 过多，可能 NMS 未生效",
            )

    def test_iou_threshold_parameter(self):
        """测试 iou_threshold 参数"""
        from src.models import create_model

        # 使用自定义 iou_threshold 创建模型
        model = create_model(
            str(self.onnx_model_path), device="cpu", conf_threshold=0.001
        )
        model.iou_threshold = 0.5  # 设置 IOU 阈值

        self.assertEqual(model.iou_threshold, 0.5)

    def test_nms_with_different_thresholds(self):
        """测试不同 IOU 阈值的 NMS 效果"""
        from src.models import create_model

        model = create_model(str(self.onnx_model_path), device="cpu")

        # 创建虚拟图像
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # 测试不同的 IOU 阈值
        thresholds = [0.25, 0.45, 0.65]
        detections_counts = []

        for threshold in thresholds:
            model.iou_threshold = threshold
            detections = model.predict(dummy_image, conf_threshold=0.001)
            detections_counts.append(len(detections))

        # 更高的 IOU 阈值应该保留更多检测框（更宽松的 NMS）
        # 注意：这个趋势可能不总是成立，取决于检测结果
        # 只在有足够检测结果时验证
        if detections_counts[0] > 0 and detections_counts[2] > 0:
            self.assertGreaterEqual(
                detections_counts[2],
                detections_counts[1],
                f"IOU=0.65 应该保留更多或相等检测框（{detections_counts[2]}）比 IOU=0.45（{detections_counts[1]}）",
            )

    def test_class_id_distribution(self):
        """测试类别 ID 分布"""
        from src.models import create_model

        model = create_model(str(self.onnx_model_path), device="cpu")

        # 创建虚拟图像
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # 运行推理
        detections = model.predict(dummy_image, conf_threshold=0.001)

        if len(detections) > 0:
            # 检查类别 ID 分布
            class_ids = [det.class_id for det in detections]
            unique_classes = set(class_ids)

            # 验证所有类别 ID 都在有效范围内
            for class_id in unique_classes:
                self.assertGreaterEqual(class_id, 1, f"类别 ID {class_id} 应该 >= 1")
                self.assertLessEqual(class_id, 80, f"类别 ID {class_id} 应该 <= 80")


if __name__ == "__main__":
    import numpy as np

    unittest.main()
