"""
模型单元测试
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from src.models import create_model, load_model_wrapper
from src.models.base import BaseModel, Detection


class TestYOLOv8:
    """YOLOv8 模型测试"""

    @pytest.fixture
    def model(self):
        """加载 YOLOv8n 模型"""
        model = create_model("yolov8n", device="cpu", conf_threshold=0.25)
        load_model_wrapper(model, "models_cache/yolov8n.pt", "YOLOv8n")
        return model

    @pytest.fixture
    def sample_image(self):
        """生成测试图片"""
        return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    def test_model_loading(self, model):
        """测试模型加载"""
        assert model is not None
        assert model.model is not None
        assert hasattr(model, "model_info")
        assert model.model_info is not None

    def test_model_info(self, model):
        """测试模型信息"""
        info = model.get_model_info()

        assert info is not None
        assert isinstance(info, dict)
        assert "name" in info
        assert "params" in info
        assert info["params"] > 0
        # 模型名称可能是 "YOLOv8" 或具体的模型名如 "YOLOv8n"
        assert "YOLOv8" in info["name"]

    def test_prediction_format(self, model, sample_image):
        """测试预测格式"""
        detections = model.predict(sample_image)

        assert isinstance(detections, list)

        for det in detections:
            assert isinstance(det, Detection)
            assert isinstance(det.bbox, list)
            assert len(det.bbox) == 4
            assert isinstance(det.confidence, float)
            assert 0 <= det.confidence <= 1
            assert isinstance(det.class_id, int)
            assert det.class_id >= 0

    def test_bbox_coordinates(self, model, sample_image):
        """测试边界框坐标有效性"""
        detections = model.predict(sample_image)

        if len(detections) > 0:
            for det in detections:
                x1, y1, x2, y2 = det.bbox

                assert x1 >= 0
                assert y1 >= 0
                assert x2 > x1
                assert y2 > y1
                assert x2 <= 640
                assert y2 <= 640

    def test_warmup(self, model, sample_image):
        """测试预热功能"""
        model.warmup((640, 640))
        assert True


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

    def test_detection_attributes(self):
        """测试 Detection 属性访问"""
        bbox = [100, 100, 200, 200]
        confidence = 0.95
        class_id = 0

        det = Detection(bbox, confidence, class_id)

        assert hasattr(det, "bbox")
        assert hasattr(det, "confidence")
        assert hasattr(det, "class_id")


class TestBaseModel:
    """BaseModel 基类测试"""

    def test_device_detection(self):
        """测试设备检测"""

        class MockModel(BaseModel):
            def load_model(self, weights_path):
                pass

            def predict(self, image):
                return []

            def get_model_info(self):
                return {}

            def warmup(self, image_size=(640, 640)):
                pass

        model = MockModel(device="auto")
        assert model.device in ["cuda", "mps", "cpu"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
