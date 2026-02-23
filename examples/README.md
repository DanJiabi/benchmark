# OD-Benchmark 示例

本目录包含 OD-Benchmark 的所有使用示例，帮助您快速掌握各项功能。

## 📋 快速索引

| 您的场景 | 推荐开始 | 预计时间 | 命令 |
|---------|---------|---------|------|
| 🆕 **首次使用** | [快速开始指南](#快速开始) | 5分钟 | `python examples/quick_start.py` |
| 📊 **测试模型性能** | [基准测试示例](#1-基准测试-benchmark) | 10分钟 | `python examples/benchmark_example.py` |
| 🔍 **对比不同模型** | [模型对比分析](#2-模型对比分析-analyze) | 10分钟 | `python examples/analyze_example.py` |
| 📤 **导出ONNX部署** | [模型导出示例](#3-模型导出-export) | 10分钟 | `python examples/export_example.py` |
| ⚖️ **PyTorch vs ONNX** | [格式对比工具](#4-格式性能对比-compare) | 15分钟 | `python examples/COMPARE_PT_ONNX.md` |
| 🚀 **ONNX性能测试** | [ONNX基准测试](#5-onnx-基准测试) | 10分钟 | `python examples/onnx_benchmark_example.py` |

## 快速开始

### 5 分钟上手

```bash
python examples/quick_start.py
```

这会打印出一个完整的快速开始指南，涵盖从安装到分析自定义模型的全过程。

## 核心功能示例

### 1. 基准测试 (benchmark)

**文件**: `examples/benchmark_example.py`

运行基准测试示例：

```bash
python examples/benchmark_example.py
```

**内容**:
- 示例 1: 测试单个模型
- 示例 2: 测试多个模型对比
- 示例 3: 测试所有配置的模型
- 示例 4: 带检测框可视化的测试
- 示例 5: 使用自定义配置文件
- 示例 6: 不同置信度阈值的影响

### 2. 模型对比分析 (analyze)

**文件**: `examples/analyze_example.py`

运行分析示例：

```bash
python examples/analyze_example.py
```

**内容**:
- 示例 1: 基础模型对比
- 示例 2: 对比所有基准模型
- 示例 3: 对比自定义权重文件
- 示例 4: 不同输出格式对比 (JSON/HTML/CSV)
- 示例 5: 调试模式

### 3. 模型导出 (export)

**文件**: `examples/export_example.py`

运行导出示例：

```bash
python examples/export_example.py
```

**内容**:
- 示例 1: 导出为 ONNX 格式
- 示例 2: 导出动态尺寸 ONNX
- 示例 3: 批量导出所有模型
- 示例 4: 不同输入尺寸导出 (320/640/1280)
- 示例 5: 批量导出脚本示例
- 示例 6: 对比 PyTorch 和 ONNX 性能

### 4. 格式性能对比 (compare)

**文件**: `examples/compare_example.py`

运行对比示例：

```bash
python examples/compare_example.py
```

**内容**:
- 示例 1: PyTorch vs ONNX 基础对比
- 示例 2: 只测试 PyTorch 格式
- 示例 3: 不同测试图片数量的影响
- 示例 4: 使用自定义模型名称
- 示例 5: 批量对比多个模型
- 示例 6: 先导出 ONNX 模型再对比
- 示例 7: 提取性能摘要

**工具**: `examples/compare_pt_onnx.py`

独立的 PyTorch vs ONNX 对比工具：

```bash
python examples/compare_pt_onnx.py --model yolov8n --num-images 50
```

**文档**: `examples/COMPARE_PT_ONNX.md`

详细的使用说明和故障排除指南。

### 5. ONNX 基准测试

**文件**: `examples/onnx_benchmark_example.py`

运行 ONNX 基准测试示例：

```bash
python examples/onnx_benchmark_example.py
```

**内容**:
- 示例 1: 导出所有模型为 ONNX 格式
- 示例 2: 批量测试所有 ONNX 模型
- 示例 3: 测试指定的 ONNX 模型
- 示例 4: 带可视化的 ONNX 模型测试
- 示例 5: 对比 PyTorch 和 ONNX 性能
- 示例 6: 一键导出并测试所有模型
- 示例 7: 不同置信度阈值测试

### 6. 结果可视化

**文件**: `examples/visualize_clean.py`

可视化检测结果：

```bash
python examples/visualize_clean.py --model yolov8n --max-images 5
```

## 命令速查

### 基准测试

```bash
# 单个模型（PyTorch）
odb benchmark --model yolov8n --num-images 10

# 多个模型
odb benchmark --model yolov8n --model yolov8s --num-images 50

# 所有模型（PyTorch）
odb benchmark --all --num-images 100

# 带可视化
odb benchmark --model yolov8n --visualize --num-viz-images 10

# 所有 ONNX 模型
odb benchmark --all --format onnx --num-images 100

# 指定 ONNX 模型
odb benchmark --model yolov8n.onnx --format onnx --num-images 50
```

### 模型分析

```bash
# 基础对比
odb analyze --baseline yolov8n --user-model yolov8s --num-images 50

# 所有基准
odb analyze --all-baselines --user-model yolov10n --num-images 100

# 自定义模型
odb analyze --baseline yolov8n --user-model path/to/model.pt --num-images 50
```

### 模型导出

```bash
# 导出 ONNX
odb export --model models_cache/yolov8n.pt --format onnx

# 动态尺寸
odb export --model models_cache/yolov8n.pt --format onnx --dynamic

# 批量导出
odb export --all-models --format onnx
```

### 格式对比

```bash
# PyTorch vs ONNX
odb compare --model models_cache/yolov8n.pt --num-images 50

# 只测试 PyTorch
odb compare --model models_cache/yolov8n.pt --formats pytorch --num-images 50
```

## 📁 文件分类

### 🎓 入门教程
| 文件 | 类型 | 描述 | 使用场景 |
|------|------|------|----------|
| `quick_start.py` | 📖 教程 | 7步完整入门指南 | **新用户首选** - 了解所有功能 |
| `check_examples_ready.py` | 🔧 工具 | 环境检查脚本 | 验证是否可运行示例 |
| `README.md` | 📖 文档 | 本文件，完整目录说明 | 查找所需示例 |

**开始之前**: 运行环境检查
```bash
python examples/check_examples_ready.py
```

### 📊 核心功能示例
| 文件 | 类型 | 描述 | 使用场景 |
|------|------|------|----------|
| `benchmark_example.py` | 💻 代码 | 6个基准测试示例 | 学习性能测试各种用法 |
| `analyze_example.py` | 💻 代码 | 5个模型对比示例 | 学习如何对比基准模型与自定义模型 |
| `export_example.py` | 💻 代码 | 6个导出示例 | 学习导出ONNX/TensorRT用于部署 |
| `compare_example.py` | 💻 代码 | 7个格式对比示例 | 学习对比PyTorch和ONNX性能 |
| `onnx_benchmark_example.py` | 💻 代码 | 7个ONNX测试示例 | 学习测试ONNX模型性能 |

### 🛠️ 实用工具
| 文件 | 类型 | 描述 | 使用场景 |
|------|------|------|----------|
| `compare_pt_onnx.py` | 🔧 工具 | PyTorch vs ONNX对比脚本 | **验证ONNX导出正确性** |
| `visualize_clean.py` | 🔧 工具 | 检测结果可视化脚本 | 查看检测框效果 |
| `COMPARE_PT_ONNX.md` | 📖 文档 | ONNX对比详细说明 | 理解格式差异和故障排除 |
| `utils.py` | 🛠️ 库 | 共享工具函数 | 示例代码内部使用 |

## 🎯 按场景选择

### 场景 1: 新用户首次使用
**推荐路径**:
1. 阅读本文档了解整体结构
2. 运行 `python examples/quick_start.py` 了解基本流程
3. 执行第一个测试: `odb benchmark --model yolov8n --num-images 10`
4. 根据需求深入学习特定功能

### 场景 2: 需要测试模型性能
**学习路径**:
1. 查看 `python examples/benchmark_example.py` 了解所有测试选项
2. 快速测试: `odb benchmark --model yolov8n --num-images 10`
3. 完整评估: `odb benchmark --model yolov8n --num-images 100 --conf-threshold 0.001`
4. 批量测试: `odb benchmark --all --num-images 100`

### 场景 3: 准备部署到生产环境
**学习路径**:
1. 查看 `python examples/export_example.py` 了解导出选项
2. 导出ONNX: `odb export --model yolov8n.pt --format onnx`
3. 阅读 `examples/COMPARE_PT_ONNX.md` 了解ONNX性能影响
4. 对比验证: `python examples/compare_pt_onnx.py --pt-model yolov8n --onnx-model models_export/yolov8n.onnx`
5. 测试ONNX性能: `odb benchmark --model yolov8n.onnx --format onnx`

### 场景 4: 对比自定义模型与基准
**学习路径**:
1. 查看 `python examples/analyze_example.py` 了解对比方法
2. 单模型对比: `odb analyze --baseline yolov8n --user-model your_model.pt`
3. 多基准对比: `odb analyze --all-baselines --user-model your_model.pt`
4. 生成HTML报告: `odb analyze ... --format html`

## 常见任务

### 任务 1: 快速测试一个模型

```bash
odb benchmark --model yolov8n --num-images 10
```

### 任务 2: 对比多个模型的性能

```bash
odb benchmark --model yolov8n --model yolov8s --model yolov10n --num-images 50
```

### 任务 3: 生成详细的可视化结果

```bash
odb benchmark --model yolov8n --visualize --num-viz-images 10 --num-images 20
```

### 任务 4: 导出模型用于部署

```bash
odb export --model models_cache/yolov8n.pt --format onnx --simplify
```

### 任务 5: 对比自定义模型与基准

```bash
odb analyze --baseline yolov8n --user-model path/to/your_model.pt --num-images 50
```

### 任务 6: 对比 PyTorch 和 ONNX 性能

```bash
odb compare --model models_cache/yolov8n.pt --num-images 100
```

### 任务 7: 批量测试所有模型

```bash
# PyTorch 模型
odb benchmark --all --num-images 100

# ONNX 模型
odb benchmark --all --format onnx --num-images 100
```

### 任务 8: 生成 HTML 报告

```bash
odb analyze --baseline yolov8n --user-model yolov8s --format html --num-images 50
```

## 📂 文件结构

### examples/ 目录结构

```
examples/
├── 📖 README.md                    # 📘 本文档 - 完整示例指南
├── 📖 COMPARE_PT_ONNX.md          # 📘 ONNX对比详细文档
│
├── 🎓 入门教程
│   ├── quick_start.py              # ⚡ 5分钟快速入门
│   └── check_examples_ready.py     # ✔️ 环境检查工具
│
├── 📊 核心功能示例
│   ├── benchmark_example.py        # 📈 基准测试示例 (6个示例)
│   ├── analyze_example.py          # 🔍 模型对比分析 (5个示例)
│   ├── export_example.py           # 📤 模型导出示例 (6个示例)
│   ├── compare_example.py          # ⚖️ 格式对比示例 (7个示例)
│   └── onnx_benchmark_example.py   # 🚀 ONNX测试示例 (7个示例)
│
├── 🛠️ 实用工具
│   ├── compare_pt_onnx.py          # 🔬 PyTorch vs ONNX对比工具
│   ├── visualize_clean.py          # 🎨 检测结果可视化
│   └── utils.py                    # 🛠️ 共享工具函数
│
└── 📁 __pycache__/                 # Python缓存 (自动生成)
```

### 输出目录结构

运行示例后，输出文件将保存在以下目录：

```
outputs/
├── examples/                  # 示例脚本输出
│   ├── single_model/          # benchmark_example.py 示例1
│   ├── multi_model/           # benchmark_example.py 示例2
│   ├── all_models/            # benchmark_example.py 示例3
│   ├── with_viz/              # benchmark_example.py 示例4
│   ├── compare/               # compare_example.py 输出
│   ├── analysis/              # analyze_example.py 输出
│   └── export/                # export_example.py 输出
│
├── results/                   # 主要基准测试结果
│   ├── comparison.json        # 批量测试结果
│   └── results_table.csv      # 对比表格
│
├── visualizations/            # 检测框可视化图片
│   └── *.jpg                  # 带检测框的示例图片
│
├── format_comparison/         # odb compare 输出
│   └── *.onnx                 # 临时导出的ONNX模型
│
├── pytorch_results/           # PyTorch格式批量测试
├── onnx_results/              # ONNX格式批量测试
│
└── figures/                   # 性能对比图表
    ├── metrics_comparison.png # 指标对比图
    ├── fps_vs_map.png         # FPS vs mAP图
    └── size_vs_performance.png # 模型大小vs性能图
```

## 进阶用法

### 批量脚本示例

创建一个批量测试脚本 `batch_test.sh`:

```bash
#!/bin/bash

MODELS=("yolov8n" "yolov8s" "yolov10n")
NUM_IMAGES=50

for model in "${MODELS[@]}"; do
    echo "测试模型: $model"
    odb benchmark --model "$model" --num-images "$NUM_IMAGES"
done

echo "所有测试完成！"
```

### Python API 使用

```python
from src.benchmark import BenchmarkRunner

# 创建测试运行器
runner = BenchmarkRunner(
    config_file="config.yaml",
    output_dir="outputs/my_test"
)

# 加载数据集
dataset = runner.load_dataset()

# 运行测试
results = runner.run_benchmark(
    models=["yolov8n", "yolov8s"],
    dataset=dataset,
    num_images=50
)

# 保存结果
runner.save_results(results)
```

## 常见问题

### Q: 如何更改数据集路径？

在 `config.yaml` 中修改 `dataset_path`:

```yaml
dataset:
  path: ~/raw/COCO  # 改为你的数据集路径
  split: val2017
```

### Q: Apple Silicon (M1/M2/M3/M4) 如何使用？

设置环境变量：

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
odb benchmark --model yolov8n --num-images 10
```

### Q: 如何添加自定义模型？

参考文档: `docs/ADD_CUSTOM_MODEL.md`

### Q: 如何理解输出结果？

- **mAP@0.50**: IoU 阈值 0.50 时的平均精度
- **mAP@0.50:0.95**: COCO 主指标，IoU 0.50-0.95 的平均 mAP
- **FPS**: 每秒帧数，越高越好
- **参数量**: 模型大小（百万参数），越小越好

## 相关文档

- `docs/ADD_CUSTOM_MODEL.md` - 添加自定义模型
- `docs/ANALYSIS_USAGE.md` - 分析功能使用
- `docs/EXPORT_GUIDE.md` - 导出指南
- `docs/FORMAT_COMPARISON.md` - 格式对比详细说明
- `docs/ONNX_FIX.md` - ONNX 后处理修复说明

## 运行所有示例

```bash
# 运行所有示例（只打印命令，不执行）
python examples/quick_start.py
python examples/benchmark_example.py
python examples/analyze_example.py
python examples/export_example.py
```

## 反馈与支持

如有问题或建议，请：
1. 查看文档目录 `docs/`
2. 查看 GitHub Issues
3. 提交新的 Issue

---

**提示**: 示例代码中只打印命令，不实际执行。您可以复制命令到终端中运行。
