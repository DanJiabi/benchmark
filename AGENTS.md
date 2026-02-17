# OD-Benchmark 开发指南

## 项目概述

OD-Benchmark 是一个目标检测模型性能基准测试工具，用于评估和对比不同目标检测模型在 COCO 数据集上的性能表现。

**核心功能**:
- 支持 YOLO 系列 (v8/v9/v10/v11)、Faster R-CNN、RT-DETR 等模型
- COCO 标准指标评估 (mAP@0.50, mAP@0.50:0.95)
- 推理性能测试 (FPS, 延迟)
- 模型导出 (ONNX, TensorRT)
- 多格式性能对比 (PyTorch vs ONNX)

## 技术栈与规范

### Python 版本
- **最低版本**: Python 3.9+
- **推荐版本**: Python 3.11
- **使用特性**: 类型提示、dataclass、pathlib

### 项目结构
```
benchmark/
├── src/                        # 主包
│   ├── models/                 # 模型实现
│   │   ├── base.py            # 基类定义
│   │   ├── ultralytics_wrapper.py
│   │   ├── onnx_model.py
│   │   ├── faster_rcnn.py
│   │   └── exporters.py       # 模型导出
│   ├── data/                   # 数据加载
│   │   └── coco_dataset.py
│   ├── metrics/                # 指标计算
│   │   └── coco_metrics.py
│   ├── analysis/               # 分析工具
│   │   ├── model_comparison.py
│   │   └── format_comparison.py
│   ├── utils/                  # 工具函数
│   │   ├── logger.py
│   │   └── visualization.py
│   └── cli.py                  # 命令行入口
├── tests/                      # 测试目录
├── scripts/                    # 辅助脚本
├── examples/                   # 示例代码
├── docs/                       # 文档
├── models_cache/               # 模型权重 (gitignored)
├── outputs/                    # 输出结果 (gitignored)
├── pyproject.toml              # 项目配置
├── config.yaml                 # 基准测试配置
├── environment.yml             # Conda 环境配置
└── run_benchmark.sh            # 启动脚本
```

### 代码风格
- **格式化**: Ruff（替代 black + isort + flake8）
- **类型检查**: mypy
- **导入排序**: Ruff 内置 isort 规则
- **行长度**: 88 字符
- **引号**: 双引号

### 虚拟环境管理
- **工具**: mamba（已安装于 ~/miniforge3）
- **环境名**: benchmark（与项目同名）
- **环境文件**: environment.yml

## 开发工作流程

### 1. 环境设置
```bash
# 创建环境
mamba env create -f environment.yml

# 激活环境
mamba activate benchmark

# 开发安装
pip install -e ".[dev]"
```

### 2. 代码规范检查
每次提交前必须运行：
```bash
# 代码格式化
ruff format .
ruff check . --fix

# 类型检查
mypy src/

# 运行测试
pytest
```

### 3. 模块职责

| 模块 | 职责 |
|------|------|
| `models/` | 模型封装、推理、导出 |
| `data/` | COCO 数据集加载 |
| `metrics/` | mAP、FPS 等指标计算 |
| `analysis/` | 模型对比、格式对比 |
| `utils/` | 日志、可视化工具 |
| `cli.py` | 命令行接口 |

## 代码规范

### 导入规范
```python
"""模块文档字符串."""

from __future__ import annotations

# 1. 标准库
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

# 2. 第三方库
import numpy as np
import torch
from tqdm import tqdm

# 3. 本地模块
from src.models.base import BaseModel, Detection
from src.utils.logger import setup_logger

if TYPE_CHECKING:
    from PIL import Image
```

### 类型提示要求
- **必须**: 所有公共函数的参数和返回值
- **推荐**: 类属性类型
- **工具**: 使用 `from __future__ import annotations` 延迟评估

### 错误处理
```python
# 自定义异常层次
class BenchmarkError(Exception):
    """基准测试基础异常."""
    pass

class ModelLoadError(BenchmarkError):
    """模型加载异常."""
    pass

class DatasetError(BenchmarkError):
    """数据集异常."""
    pass

# 使用示例
try:
    model = create_model(model_name)
except ValueError as e:
    logger.error(f"不支持的模型类型: {model_name}")
    logger.error(f"错误: {e}")
    return None
except FileNotFoundError as e:
    logger.error(f"模型文件不存在: {weights_path}")
    raise ModelLoadError(f"无法加载模型: {e}") from e
```

### 日志记录
```python
from src.utils.logger import setup_logger

# 使用 rich logger
logger = setup_logger(config)

logger.info("开始评估模型: {model_name}")
logger.info(f"模型信息: {model_info}")
logger.error(f"推理失败 (图片 {idx}): {e}")
logger.warning("继续执行，但首次推理可能较慢")
```

## 模型规范

### 基类继承
```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np

class BaseModel(ABC):
    """模型基类."""
    
    def __init__(self, device: str = "auto", conf_threshold: float = 0.001):
        self.device = self._get_device(device)
        self.conf_threshold = conf_threshold
        self.model = None
        self.model_info = {}
    
    @abstractmethod
    def load_model(self, weights_path: str) -> None:
        """加载模型权重."""
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> List[Detection]:
        """执行推理."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息."""
        pass
    
    @abstractmethod
    def warmup(self, image_size: tuple = (640, 640)) -> None:
        """模型预热."""
        pass
```

### 设备选择
```python
def _get_device(self, device: str) -> str:
    if device != "auto":
        return device
    
    import torch
    
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
```

## 测试规范

### 测试结构
```python
# tests/test_models.py
import pytest
import numpy as np
from src.models import create_model

class TestModelCreation:
    """模型创建测试."""
    
    def test_create_yolov8n(self) -> None:
        """测试创建 YOLOv8n 模型."""
        model = create_model("yolov8n", device="cpu")
        assert model is not None
        assert model.device == "cpu"
    
    @pytest.mark.slow
    def test_model_inference(self) -> None:
        """测试模型推理（慢速测试）."""
        model = create_model("yolov8n", device="cpu")
        model.load_model(None)  # 使用预训练权重
        
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        detections = model.predict(dummy_image)
        
        assert isinstance(detections, list)
    
    @pytest.mark.parametrize("model_name", ["yolov8n", "yolov8s"])
    def test_multiple_models(self, model_name: str) -> None:
        """参数化测试多个模型."""
        model = create_model(model_name, device="cpu")
        assert model is not None
```

### 测试标记
```python
# pytest.ini_options 中定义的标记
@pytest.mark.slow          # 慢速测试
@pytest.mark.integration   # 集成测试
@pytest.mark.skip_ci       # 跳过 CI
```

### 测试要求
- **覆盖率**: 核心模块 >= 70%
- **fixtures**: 使用 `conftest.py` 共享
- **mock**: 使用 `pytest-mock` 模拟外部依赖

## 数据管理

### COCO 数据集结构
```
~/raw/COC/
├── annotations/
│   ├── instances_val2017.json
│   └── ...
├── val2017/
│   ├── 000000000009.jpg
│   └── ...
└── ...
```

### 模型权重目录
```
models_cache/
├── yolov8n.pt
├── yolov8s.pt
├── yolov8m.pt
└── ...        # 所有 .pt 文件被 gitignore
```

### 输出结果
```
outputs/
├── results/           # 基准测试结果
│   ├── yolov8n_result.json
│   └── comparison.json
├── figures/           # 图表
│   ├── metrics_comparison.png
│   └── fps_vs_map.png
├── visualizations/    # 可视化图片
├── analysis/          # 分析报告
└── logs/              # 日志文件
```

## 配置文件

### pyproject.toml 关键配置
```toml
[project]
name = "od-benchmark"
version = "0.1.0"
requires-python = ">=3.9"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "mypy>=1.0",
    "ruff>=0.1",
]

[tool.ruff]
line-length = 88
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow",
    "integration: marks tests as integration tests",
]
```

### config.yaml 说明
```yaml
dataset:
  path: ~/raw/COCO          # COCO 数据集路径
  split: val2017            # 验证集

models:                     # 待测试模型列表
  - name: yolov8n
    framework: ultralytics
    weights: yolov8n.pt
    url: https://...        # 自动下载地址

evaluation:
  device: auto              # cuda/mps/cpu
  conf_threshold: 0.25      # 置信度阈值
  iou_threshold: 0.6        # NMS IoU 阈值
```

## 性能优化

### 硬件加速
- **优先级**: CUDA > MPS (Apple Silicon) > CPU
- **自动检测**: 使用 `_get_device("auto")`
- **MPS 注意**: 需设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`

### 批处理推理
```python
# 对于大批量数据，使用批处理提高效率
for batch_images in DataLoader(dataset, batch_size=32):
    batch_predictions = model.predict_batch(batch_images)
```

### 模型导出优化
```bash
# ONNX 导出（推荐用于部署）
od-benchmark export --model model.pt --format onnx --simplify

# TensorRT 导出（NVIDIA GPU 最优）
od-benchmark export --model model.pt --format tensorrt --fp16
```

## 常见运行问题

### Apple Silicon MPS 兼容性

**问题**: `torchvision::nms` 操作不支持 MPS 后端

**解决方案**:
```bash
# 方式 1：使用启动脚本（推荐）
./run_benchmark.sh --config config.yaml --model yolov8n

# 方式 2：手动设置环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1
python benchmark.py --config config.yaml --model yolov8n
```

### 相对导入错误

**问题**: `ImportError: attempted relative import beyond top-level package`

**解决方案**: 从项目根目录运行
```bash
cd /Users/danjiabi/github/benchmark
python benchmark.py --config config.yaml --model yolov8n
```

### 模型下载失败

**问题**: GitHub 下载超时或失败

**解决方案**:
```bash
# 手动下载模型权重到 models_cache/
wget -O models_cache/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### 内存不足

**问题**: 大模型推理时 OOM

**解决方案**:
```bash
# 减少测试图片数量
python benchmark.py --config config.yaml --model yolov8x --num-images 100

# 使用 CPU 推理（慢但稳定）
export PYTORCH_ENABLE_MPS_FALLBACK=1
python benchmark.py --config config.yaml --model yolov8x --device cpu
```

## Git 工作流程

### 提交消息格式
- 遵循 Conventional Commits 格式：`type: subject`
- 类型：`init`, `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
- 示例：`feat: 添加 yolov11 模型支持`

### .gitignore 配置
```gitignore
# 模型权重
*.pt
*.pth
*.onnx
*.engine

# 输出目录
outputs/
models_cache/

# 临时文件
.temp/
__pycache__/
```

## 调试与开发

### 临时脚本
- **位置**: `.temp/` 目录（已 gitignored）
- **命名**: `test_*.py`, `debug_*.py`
- **清理**: 定期清理，不提交到版本控制

### 常用调试命令
```bash
# 快速测试（少量图片）
python benchmark.py --config config.yaml --model yolov8n --num-images 10

# 启用可视化
python benchmark.py --config config.yaml --model yolov8n --visualize --num-viz-images 5

# 分析模式
od-benchmark analyze --baseline yolov8n --user-model path/to/model.pt --num-images 50
```

## 安全注意事项

- **模型权重**: 不要将大型 .pt 文件提交到 git
- **数据集路径**: 使用 `~` 展开用户目录，避免硬编码
- **API 密钥**: 不要在代码中硬编码任何密钥
- **敏感数据**: 确保 .gitignore 正确配置

## 文档规范

### 代码文档
```python
def run_single_model(
    model_config: Dict[str, Any],
    dataset: COCOInferenceDataset,
    coco_metrics_calculator: COCOMetrics,
    logger,
    max_images: Optional[int] = None,
    conf_threshold: float = 0.001,
) -> Optional[Dict[str, Any]]:
    """运行单个模型的基准测试.
    
    Args:
        model_config: 模型配置字典，包含 name, framework, weights 等字段
        dataset: COCO 推理数据集
        coco_metrics_calculator: COCO 指标计算器
        logger: 日志记录器
        max_images: 最大测试图片数量，None 表示全部
        conf_threshold: 置信度阈值
        
    Returns:
        包含 coco_metrics, performance, model_info 的结果字典，
        如果测试失败则返回 None
        
    Example:
        >>> result = run_single_model(
        ...     {"name": "yolov8n", "framework": "ultralytics"},
        ...     dataset, metrics, logger
        ... )
        >>> print(result["coco_metrics"]["AP@0.50"])
        0.3730
    """
```

## Obsidian 集成规则

在条件允许的情况下，使用 Obsidian 相关的 skills 进行项目文档管理：

1. **项目文件夹创建**:
   - 在本地 Obsidian Vault 中创建与项目同名的文件夹（`benchmark`）
   - 使用 `obsidian-cli` skill 进行操作

2. **文档同步规则**:
   - 将项目中的文档（README、开发规范、设计文档等）同步至 Obsidian
   - 使用 Obsidian Markdown 格式（wikilinks、callouts、properties）
   - 保持项目和 Obsidian 文档的实时同步

3. **同步触发时机**:
   - 每次更新文档后自动同步
   - 使用 obsidian-cli 的 note create/update 命令
   - 保持文档结构和格式的一致性

4. **示例操作**:
   ```bash
   # 使用 obsidian-cli 创建/更新笔记
   obsidian-cli note create "benchmark/开发规范" --content "..."
   obsidian-cli note update "benchmark/README" --content-file README.md
   ```

## 注意事项

1. **禁止**: 不要将大型模型权重文件提交到仓库
2. **注意**: Apple Silicon 需要设置 MPS fallback 环境变量
3. **推荐**: 使用 `./run_benchmark.sh` 启动脚本运行基准测试
4. **必须**: 所有公共函数都要有类型提示
5. **必须**: 提交前运行 `ruff check .` 和 `pytest`
6. **建议**: 使用 `--num-images` 参数进行快速验证

---

**开发前必读此文档，确保代码风格一致。**
