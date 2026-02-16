"""
数据集单元测试
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from pathlib import Path

from src.data.coco_dataset import COCOInferenceDataset


class TestCOCOInferenceDataset:
    """COCO 数据集测试"""

    @pytest.fixture
    def dataset_path(self):
        """数据集路径"""
        return "~/raw/COCO"

    @pytest.fixture
    def dataset(self, dataset_path):
        """创建数据集对象"""
        expanded_path = Path(dataset_path).expanduser()
        if not expanded_path.exists():
            pytest.skip(f"COCO 数据集不存在: {expanded_path}")
        return COCOInferenceDataset(dataset_path, "val2017")

    def test_dataset_initialization(self, dataset):
        """测试数据集初始化"""
        assert dataset is not None
        assert hasattr(dataset, "dataset_path")
        assert hasattr(dataset, "split")
        assert dataset.split == "val2017"

    def test_dataset_length(self, dataset):
        """测试数据集长度"""
        length = len(dataset)
        assert length > 0
        # COCO val2017 应该有 5000 张图片
        assert length == 5000

    def test_getitem(self, dataset):
        """测试数据获取"""
        image_id, image = dataset[0]

        assert isinstance(image_id, int)
        assert image_id > 0
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3
        assert image.shape[2] == 3  # RGB/BGR

    def test_image_format(self, dataset):
        """测试图片格式"""
        _, image = dataset[0]

        assert image.dtype == np.uint8
        assert image.shape[2] == 3
        # 检查像素值范围
        assert image.min() >= 0
        assert image.max() <= 255

    def test_multiple_access(self, dataset):
        """测试多次访问同一数据"""
        image_id1, image1 = dataset[0]
        image_id2, image2 = dataset[0]

        assert image_id1 == image_id2
        assert np.array_equal(image1, image2)

    def test_different_samples(self, dataset):
        """测试不同样本"""
        image_id1, image1 = dataset[0]
        image_id2, image2 = dataset[1]

        assert image_id1 != image_id2
        # 图片内容应该不同
        assert not np.array_equal(image1, image2)

    def test_image_sizes(self, dataset):
        """测试图片尺寸"""
        # COCO 图片尺寸不一
        _, image1 = dataset[0]
        _, image2 = dataset[100]

        # 至少有一个维度不同
        assert image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1]


class TestCOCOPathHandling:
    """COCO 路径处理测试"""

    def test_path_expansion(self):
        """测试路径展开"""
        dataset = COCOInferenceDataset("~/raw/COCO", "val2017")
        expanded = Path(dataset.dataset_path).expanduser()
        assert "~" not in str(expanded)

    def test_invalid_path(self):
        """测试无效路径"""
        with pytest.raises((FileNotFoundError, Exception)):
            dataset = COCOInferenceDataset("/invalid/path", "val2017")
            len(dataset)  # 触发加载


class TestCOCOAnnotations:
    """COCO 标注测试"""

    @pytest.fixture
    def annotations_file(self):
        """标注文件路径"""
        return "~/raw/COCO/annotations/instances_val2017.json"

    def test_annotations_exist(self, annotations_file):
        """测试标注文件存在"""
        path = Path(annotations_file).expanduser()
        if not path.exists():
            pytest.skip(f"标注文件不存在: {path}")
        assert path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
