"""
ONNX 模型单元测试
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from pathlib import Path

from src.models.onnx_model import ONNXModel
from src.models import create_model, load_model_wrapper


class TestONNXModel:
    """ONNX 模型测试"""

    @pytest.fixture
    def onnx_path(self):
        """ONNX 模型路径"""
        return "models_export/yolov10n.onnx"

    @pytest.fixture
    def model(self, onnx_path):
        """加载 ONNX 模型"""
        if not Path(onnx_path).exists():
            pytest.skip(f"ONNX 模型不存在: {onnx_path}")
        model = create_model(onnx_path, device="cpu", conf_threshold=0.25)
        load_model_wrapper(model, onnx_path, onnx_path)
        return model

    @pytest.fixture
    def sample_image(self):
        """生成测试图片"""
        return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    def test_model_loading(self, model):
        """测试 ONNX 模型加载"""
        assert model is not None
        assert model.session is not None
        assert hasattr(model, "input_name")
        assert hasattr(model, "output_names")

    def test_model_info(self, model):
        """测试模型信息"""
        info = model.get_model_info()
        assert info is not None
        assert isinstance(info, dict)
        assert "name" in info
        assert "format" in info
        assert info["format"] == "onnx"
        assert "input_shape" in info

    def test_input_shape(self, model):
        """测试输入形状"""
        assert model.input_shape is not None
        assert len(model.input_shape) == 4
        assert model.input_shape[0] == 1  # batch
        assert model.input_shape[1] == 3  # channels

    def test_preprocessing(self, model, sample_image):
        """测试预处理"""
        input_tensor = model._preprocess(sample_image)
        assert input_tensor is not None
        assert len(input_tensor.shape) == 4
        assert input_tensor.shape[0] == 1  # batch
        assert input_tensor.shape[1] == 3  # channels
        assert input_tensor.dtype == np.float32

    def test_prediction_format(self, model, sample_image):
        """测试预测格式"""
        detections = model.predict(sample_image, conf_threshold=0.25)
        assert isinstance(detections, list)

        for det in detections:
            assert hasattr(det, "bbox")
            assert hasattr(det, "confidence")
            assert hasattr(det, "class_id")
            assert isinstance(det.bbox, list)
            assert len(det.bbox) == 4
            assert isinstance(det.confidence, float)
            assert isinstance(det.class_id, int)

    def test_bbox_coordinates_valid(self, model, sample_image):
        """测试边界框坐标有效性"""
        detections = model.predict(sample_image, conf_threshold=0.1)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            assert x1 >= 0
            assert y1 >= 0
            assert x2 > x1
            assert y2 > y1

    def test_class_names_loading(self, model):
        """测试类别名称加载"""
        assert hasattr(model, "names")
        assert isinstance(model.names, dict)
        # 应该有 80 个 COCO 类别
        assert len(model.names) == 80

    def test_warmup(self, model):
        """测试预热功能"""
        model.warmup((640, 640))
        assert True  # 如果没有异常则通过


class TestONNXPostprocess:
    """ONNX 后处理测试"""

    def test_format_detection_yolov10(self):
        """测试 YOLOv10 格式检测"""
        onnx_path = "models_export/yolov10n.onnx"
        if not Path(onnx_path).exists():
            pytest.skip(f"ONNX 模型不存在: {onnx_path}")

        model = create_model(onnx_path, device="cpu", conf_threshold=0.25)
        load_model_wrapper(model, onnx_path, onnx_path)

        info = model.get_model_info()
        output_shape = info.get("outputs", [{}])[0].get("shape", [])

        # YOLOv10 输出应该是 [1, num_predictions, 6]
        if len(output_shape) >= 2:
            assert output_shape[2] == 6  # 6 = [x1, y1, x2, y2, conf, class_id]

    def test_format_detection_yolov8(self):
        """测试 YOLOv8 格式检测"""
        onnx_path = "models_export/yolov8n.onnx"
        if not Path(onnx_path).exists():
            pytest.skip(f"ONNX 模型不存在: {onnx_path}")

        model = create_model(onnx_path, device="cpu", conf_threshold=0.25)
        load_model_wrapper(model, onnx_path, onnx_path)

        info = model.get_model_info()
        output_shape = info.get("outputs", [{}])[0].get("shape", [])

        # YOLOv8 输出应该是 [1, 84, num_anchors]
        if len(output_shape) >= 2:
            assert output_shape[1] == 84  # 84 = 4 (bbox) + 80 (classes)


class TestONNXModelInputSize:
    """ONNX 模型输入尺寸测试"""

    def test_default_input_size(self):
        """测试默认输入尺寸"""
        model = ONNXModel(device="cpu")
        h, w = model._get_model_input_size()
        assert h == 640
        assert w == 640

    def test_custom_input_size(self):
        """测试自定义输入尺寸"""
        onnx_path = "models_export/yolov9t.onnx"
        if not Path(onnx_path).exists():
            pytest.skip(f"ONNX 模型不存在: {onnx_path}")

        model = create_model(onnx_path, device="cpu")
        load_model_wrapper(model, onnx_path, onnx_path)

        h, w = model._get_model_input_size()
        # YOLOv9t 使用 320x320
        assert h == 320
        assert w == 320


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
