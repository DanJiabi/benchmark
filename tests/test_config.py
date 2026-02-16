"""
配置单元测试
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path

from src.utils.logger import Config


class TestConfig:
    """配置测试"""

    @pytest.fixture
    def config(self):
        """加载默认配置"""
        config_path = Path(__file__).parent.parent / "config.yaml"
        if not config_path.exists():
            pytest.skip(f"配置文件不存在: {config_path}")
        return Config(str(config_path))

    def test_config_loading(self, config):
        """测试配置加载"""
        assert config is not None
        assert config.config is not None
        assert isinstance(config.config, dict)

    def test_models_config(self, config):
        """测试模型配置"""
        models = config.get_models_config()
        assert models is not None
        assert isinstance(models, list)
        assert len(models) > 0

        # 检查第一个模型的配置
        first_model = models[0]
        assert "name" in first_model
        assert "framework" in first_model

    def test_dataset_config(self, config):
        """测试数据集配置"""
        dataset_config = config.get_dataset_config()
        assert dataset_config is not None
        assert "path" in dataset_config
        assert "split" in dataset_config

    def test_evaluation_config(self, config):
        """测试评估配置"""
        eval_config = config.get_evaluation_config()
        assert eval_config is not None

    def test_config_has_key(self, config):
        """测试配置键存在检查"""
        assert config.has("models")
        assert config.has("dataset")
        assert config.has("evaluation")
        assert not config.has("non_existent_key")

    def test_config_get(self, config):
        """测试配置获取"""
        models = config.get("models")
        assert models is not None

        # 测试默认值
        default = config.get("non_existent", "default_value")
        assert default == "default_value"

    def test_output_dir(self, config):
        """测试输出目录配置"""
        output_dir = config.config.get("output_dir")
        assert output_dir is not None
        assert isinstance(output_dir, str)


class TestConfigErrorHandling:
    """配置错误处理测试"""

    def test_nonexistent_config(self):
        """测试不存在的配置文件"""
        with pytest.raises(FileNotFoundError):
            Config("/nonexistent/path/config.yaml")

    def test_invalid_config(self, tmp_path):
        """测试无效的配置文件"""
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content:[")

        with pytest.raises(Exception):
            Config(str(invalid_config))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
