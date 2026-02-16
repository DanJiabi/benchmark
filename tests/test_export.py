"""
模型导出单元测试
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from pathlib import Path
import tempfile

from src.models.exporters import ExportManager


class TestExportManager:
    """导出管理器测试"""

    def test_export_manager_initialization(self):
        """测试导出管理器初始化"""
        manager = ExportManager()
        assert manager is not None


class TestONNXExport:
    """ONNX 导出测试"""

    def test_export_manager_exists(self):
        """测试 ExportManager 类存在"""
        assert ExportManager is not None
        assert hasattr(ExportManager, "export_to_onnx")
        assert hasattr(ExportManager, "export_to_tensorrt")
        assert hasattr(ExportManager, "export_all")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
