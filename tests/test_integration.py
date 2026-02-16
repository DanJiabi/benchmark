"""
集成测试 - 端到端测试
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
from pathlib import Path

from src.models import create_model, load_model_wrapper
from src.data.coco_dataset import COCOInferenceDataset
from src.metrics.coco_metrics import PerformanceMetrics, COCOMetrics
from src.utils.logger import Config


class TestEndToEndPipeline:
    """端到端流水线测试"""

    @pytest.fixture
    def config(self):
        """加载配置"""
        config_path = Path(__file__).parent.parent / "config.yaml"
        return Config(str(config_path))

    def test_full_benchmark_pipeline(self, config):
        """测试完整基准测试流水线"""
        # 1. 加载配置
        models_config = config.get_models_config()
        assert len(models_config) > 0

        # 2. 获取模型名称
        model_name = models_config[0]["name"]
        assert model_name is not None

        # 3. 检查模型权重文件是否存在
        weights_path = Path("models_cache") / f"{model_name}.pt"
        if not weights_path.exists():
            pytest.skip(f"模型权重不存在: {weights_path}")

        # 4. 加载模型
        model = create_model(model_name, device="cpu", conf_threshold=0.25)
        load_model_wrapper(model, str(weights_path), model_name)

        # 5. 验证模型加载成功
        assert model is not None
        info = model.get_model_info()
        assert info is not None

    def test_config_to_model_loading(self):
        """测试从配置到模型加载的流程"""
        # 从配置读取
        config = Config("config.yaml")
        models = config.get_models_config()

        # 选择一个模型
        model_config = models[0]
        model_name = model_config["name"]

        # 创建模型
        model = create_model(model_name, device="cpu")
        assert model is not None

    def test_dataset_loading(self):
        """测试数据集加载"""
        dataset_path = "~/raw/COCO"
        expanded_path = Path(dataset_path).expanduser()

        if not expanded_path.exists():
            pytest.skip(f"COCO 数据集不存在: {expanded_path}")

        dataset = COCOInferenceDataset(dataset_path, "val2017")

        # 验证数据集
        assert len(dataset) > 0
        assert len(dataset) == 5000  # COCO val2017

        # 获取样本
        image_id, image = dataset[0]
        assert image_id > 0
        assert image is not None
        assert len(image.shape) == 3

    def test_performance_metrics_collection(self):
        """测试性能指标收集"""
        metrics = PerformanceMetrics()

        # 模拟多次推理
        for _ in range(5):
            start = metrics.start_timer()
            # 模拟推理时间
            import time

            time.sleep(0.001)
            elapsed = metrics.end_timer(start)
            metrics.add_inference_time(elapsed)

        # 验证统计
        assert metrics.total_images == 5
        stats = metrics.compute_performance_stats()
        assert stats["total_images"] == 5
        assert stats["fps"] > 0


class TestModelInferenceIntegration:
    """模型推理集成测试"""

    def test_yolov8_inference(self):
        """测试 YOLOv8 推理"""
        model_path = "models_cache/yolov8n.pt"
        if not Path(model_path).exists():
            pytest.skip(f"模型不存在: {model_path}")

        # 加载模型
        model = create_model("yolov8n", device="cpu")
        load_model_wrapper(model, model_path, "yolov8n")

        # 创建测试图片
        import numpy as np

        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # 推理
        detections = model.predict(image)

        # 验证输出格式
        assert isinstance(detections, list)
        for det in detections:
            assert hasattr(det, "bbox")
            assert hasattr(det, "confidence")
            assert hasattr(det, "class_id")

    def test_onnx_inference(self):
        """测试 ONNX 推理"""
        onnx_path = "models_export/yolov10n.onnx"
        if not Path(onnx_path).exists():
            pytest.skip(f"ONNX 模型不存在: {onnx_path}")

        # 加载 ONNX 模型
        model = create_model(onnx_path, device="cpu")
        load_model_wrapper(model, onnx_path, onnx_path)

        # 创建测试图片
        import numpy as np

        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # 推理
        detections = model.predict(image)

        # 验证输出格式
        assert isinstance(detections, list)


class TestCLIIntegration:
    """CLI 集成测试"""

    def test_cli_help(self):
        """测试 CLI 帮助"""
        import subprocess

        result = subprocess.run(
            ["conda", "run", "-n", "benchmark", "od-benchmark", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout

    def test_benchmark_subcommand_help(self):
        """测试 benchmark 子命令帮助"""
        import subprocess

        result = subprocess.run(
            ["conda", "run", "-n", "benchmark", "od-benchmark", "benchmark", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--model" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
