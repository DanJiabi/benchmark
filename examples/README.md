# OD-Benchmark 示例

本目录包含 OD-Benchmark 的所有使用示例，帮助您快速掌握各项功能。

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

### 5. 结果可视化

**文件**: `examples/visualize_clean.py`

可视化检测结果：

```bash
python examples/visualize_clean.py --model yolov8n --max-images 5
```

## 命令速查

### 基准测试

```bash
# 单个模型
od-benchmark benchmark --model yolov8n --num-images 10

# 多个模型
od-benchmark benchmark --model yolov8n --model yolov8s --num-images 50

# 所有模型
od-benchmark benchmark --all --num-images 100

# 带可视化
od-benchmark benchmark --model yolov8n --visualize --num-viz-images 10
```

### 模型分析

```bash
# 基础对比
od-benchmark analyze --baseline yolov8n --user-model yolov8s --num-images 50

# 所有基准
od-benchmark analyze --all-baselines --user-model yolov10n --num-images 100

# 自定义模型
od-benchmark analyze --baseline yolov8n --user-model path/to/model.pt --num-images 50
```

### 模型导出

```bash
# 导出 ONNX
od-benchmark export --model models_cache/yolov8n.pt --format onnx

# 动态尺寸
od-benchmark export --model models_cache/yolov8n.pt --format onnx --dynamic

# 批量导出
od-benchmark export --all-models --format onnx
```

### 格式对比

```bash
# PyTorch vs ONNX
od-benchmark compare --model models_cache/yolov8n.pt --num-images 50

# 只测试 PyTorch
od-benchmark compare --model models_cache/yolov8n.pt --formats pytorch --num-images 50
```

## 示例文件说明

| 文件 | 描述 | 使用场景 |
|------|------|----------|
| `quick_start.py` | 快速开始指南 | 新用户首次使用 |
| `benchmark_example.py` | 基准测试示例 | 学习如何运行性能测试 |
| `analyze_example.py` | 模型对比示例 | 学习如何对比不同模型 |
| `export_example.py` | 模型导出示例 | 学习如何导出模型部署 |
| `compare_example.py` | 格式对比示例 | 学习如何对比不同格式性能 |
| `compare_pt_onnx.py` | 格式对比工具 | 对比 PyTorch 和 ONNX 性能 |
| `visualize_clean.py` | 可视化工具 | 查看检测结果 |
| `COMPARE_PT_ONNX.md` | ONNX 对比文档 | 详细的使用说明和故障排除 |

## 常见任务

### 任务 1: 快速测试一个模型

```bash
od-benchmark benchmark --model yolov8n --num-images 10
```

### 任务 2: 对比多个模型的性能

```bash
od-benchmark benchmark --model yolov8n --model yolov8s --model yolov10n --num-images 50
```

### 任务 3: 生成详细的可视化结果

```bash
od-benchmark benchmark --model yolov8n --visualize --num-viz-images 10 --num-images 20
```

### 任务 4: 导出模型用于部署

```bash
od-benchmark export --model models_cache/yolov8n.pt --format onnx --simplify
```

### 任务 5: 对比自定义模型与基准

```bash
od-benchmark analyze --baseline yolov8n --user-model path/to/your_model.pt --num-images 50
```

### 任务 6: 对比 PyTorch 和 ONNX 性能

```bash
od-benchmark compare --model models_cache/yolov8n.pt --num-images 100
```

### 任务 7: 批量测试所有模型

```bash
od-benchmark benchmark --all --num-images 100
```

### 任务 8: 生成 HTML 报告

```bash
od-benchmark analyze --baseline yolov8n --user-model yolov8s --format html --num-images 50
```

## 输出目录结构

```
outputs/
├── examples/              # 示例输出
│   ├── single_model/      # 单个模型测试
│   ├── multi_model/       # 多个模型对比
│   ├── with_viz/          # 带可视化的测试
│   └── analysis/          # 分析报告
├── results/               # 主要测试结果
├── visualizations/        # 检测框可视化
├── format_comparison/     # 格式对比报告
└── export/               # 导出的模型
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
    od-benchmark benchmark --model "$model" --num-images "$NUM_IMAGES"
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
od-benchmark benchmark --model yolov8n --num-images 10
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
