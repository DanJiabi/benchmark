# Benchmark - Object Detection Performance Evaluation

基于 COCO 数据集的目标检测模型性能基准测试项目。

## 📚 文档导航

| 文档 | 说明 |
|------|------|
| **本文档** | 快速开始和完整功能介绍 |
| [examples/](examples/) | 使用示例和快速开始指南 |
| [docs/ADD_CUSTOM_MODEL.md](docs/ADD_CUSTOM_MODEL.md) | 添加自定义模型指南 |
| [docs/ANALYSIS_USAGE.md](docs/ANALYSIS_USAGE.md) | 模型对比分析使用指南 |
| [docs/EXPORT_GUIDE.md](docs/EXPORT_GUIDE.md) | 模型导出指南 (ONNX/TensorRT) |
| [docs/FORMAT_COMPARISON.md](docs/FORMAT_COMPARISON.md) | 格式性能对比指南 |
| [CHANGELOG.md](CHANGELOG.md) | 版本更新历史 |

## 📦 Installation

### Quick Install (Editable Mode - Recommended)

```bash
# Create virtual environment
mamba env create -f environment.yml --force

# Activate environment
conda activate benchmark

# Install in editable mode
pip install -e .

# Verify installation
python test_installation.py
```

### Traditional Installation

```bash
# Create virtual environment
mamba env create -f environment.yml --force

# Activate environment
conda activate benchmark

# No pip install needed - just run scripts directly
python benchmark.py --model yolov8n
```

## ✨ 特性

- **多模型支持**: YOLOv8 (n/s/m/l/x), YOLOv9, YOLOv10, RT-DETR, Faster R-CNN
- **完整评估**: COCO mAP, AR, FPS 等全面性能指标
- **灵活配置**: YAML 配置文件 + 丰富的命令行参数
- **可视化**: 检测框可视化、性能对比图、PR曲线
- **跨平台**: 支持 CUDA, MPS (Apple Silicon), CPU

## 🚀 快速开始

### 方式 1: 现代包安装（推荐）

```bash
# 创建虚拟环境
mamba env create -f environment.yml --force

# 激活环境
conda activate benchmark

# 以可编辑模式安装
pip install -e .

# 验证安装
python scripts/test_installation.py

# 运行基准测试
od-benchmark benchmark --model yolov8n --num-images 10
```

### 方式 2: 传统方式（无需 pip install）

```bash
# 创建虚拟环境
mamba env create -f environment.yml --force

# 激活环境
conda activate benchmark

# 直接运行脚本（无需 pip install）
python benchmark.py --model yolov8n --num-images 10
```

### 方式 3: 使用包装脚本（自动设置环境）

```bash
# 使用 run_benchmark.sh 脚本
# 自动设置 PYTORCH_ENABLE_MPS_FALLBACK 环境变量
./run_benchmark.sh --model yolov8n --num-images 10
```

### 2. 下载模型权重

模型权重会自动从 GitHub 下载并缓存到 `models_cache/` 目录。

#### 方式 1: 使用下载工具（推荐）

使用 `scripts/download_weights.py` 工具批量下载所有模型权重：

```bash
# 下载 config.yaml 中的所有模型权重
python scripts/download_weights.py

# 使用指定配置文件
python scripts/download_weights.py --config config_test.yaml

# 指定缓存目录
python scripts/download_weights.py --cache-dir /path/to/cache

# 覆盖已存在的文件
python scripts/download_weights.py --overwrite

# 仅检查文件完整性（不下载）
python scripts/download_weights.py --check-only
```

**下载工具功能**：
- 从 config.yaml 读取模型配置
- 批量下载所有权重文件
- 检查已存在文件的完整性
- 自动重新下载不完整的文件
- 显示下载进度
- 保存下载结果到文件

**文件完整性检查**：
- 文件大小验证（防止只有几字节的错误文件）
- PyTorch 模型加载验证
- 文件可读性测试

#### 方式 2: 自动下载

运行基准测试时会自动下载缺失的模型权重：

```bash
python benchmark.py --model yolov8n
```

已下载的模型：
- ✅ yolov8n.pt (6.2MB, 3.16M 参数)
- ✅ yolov8s.pt (22MB, 11.17M 参数)
- ✅ yolov8m.pt (50MB, 25.90M 参数)
- ✅ yolov8l.pt (84MB, 43.69M 参数)
- ✅ yolov8x.pt (131MB, 68.23M 参数)

### 3. 运行基准测试

```bash
# 方式 1: 使用 CLI 工具（推荐）
od-benchmark benchmark --model yolov8n --num-images 10

# 方式 2: 使用包装脚本（自动设置环境）
./run_benchmark.sh --model yolov8n --num-images 10

# 方式 3: 使用 Python 脚本（向后兼容）
export PYTORCH_ENABLE_MPS_FALLBACK=1
python benchmark.py --model yolov8n --num-images 10
```

### 4. 查看结果

```bash
# 查看结果表格
cat outputs/results/results_table.csv

# 查看可视化
open outputs/visualizations/
```

## 📋 命令行参数

### 运行方式

本项目提供三种运行方式，按推荐顺序排列：

#### 方式 1: 使用 CLI 工具（推荐）

```bash
# 直接使用 CLI 工具
od-benchmark benchmark [options]
```

优点：
- 现代化的命令行界面
- 统一的参数处理
- 更好的错误信息

#### 方式 2: 使用包装脚本（自动设置环境）

```bash
# 使用包装脚本（自动设置 PYTORCH_ENABLE_MPS_FALLBACK）
./run_benchmark.sh [options]
```

优点：
- 自动设置环境变量
- 自动检测 Python 命令
- 跨平台兼容

#### 方式 3: 使用 Python 脚本（向后兼容）

```bash
# 直接运行 Python 脚本
export PYTORCH_ENABLE_MPS_FALLBACK=1
python benchmark.py [options]
```

说明：
- 保留原有的使用方式
- 需要手动设置环境变量
- 适合习惯直接运行 Python 脚本的用户

### 基础参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | config.yaml |
| `--model` | 指定要测试的模型（可多次使用） | - |
| `--all` | 测试所有配置的模型 | False |
| `--output-dir` | 输出目录 | outputs/results |

### 可视化参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--visualize` | 启用检测框可视化 | False |
| `--num-viz-images` | 可视化图片数量 | 10 |

### 性能参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--conf-threshold` | 置信度阈值 | 使用配置文件中的值 |

### 示例

```bash
# 使用低阈值进行完整mAP评估（推荐）
python benchmark.py --all --conf-threshold 0.001

# 使用高阈值生成清晰的可视化
python benchmark.py --model yolov8n --visualize --conf-threshold 0.25

# 对比不同阈值的影响
for conf in 0.001 0.01 0.05 0.1 0.25; do
  python benchmark.py --model yolov8n --conf-threshold $conf
done
```

## 📁 项目结构

```
benchmark/
├── environment.yml          # Conda 环境配置
├── config.yaml              # 主配置文件
├── config_test.yaml         # 测试配置文件
├── requirements.txt         # Python 依赖
├── pyproject.toml          # Python 包配置
├── benchmark.py            # 主运行脚本
├── scripts/               # 工具脚本
│   ├── download_weights.py  # 权重下载工具
│   └── test_installation.py # 安装验证脚本
├── examples/               # 示例代码和文档
│   ├── quick_start.py       # 快速开始指南
│   ├── benchmark_example.py # 基准测试示例
│   ├── analyze_example.py   # 模型对比示例
│   ├── export_example.py    # 模型导出示例
│   ├── compare_example.py   # 格式对比示例
│   ├── compare_pt_onnx.py   # PyTorch vs ONNX 对比工具
│   ├── visualize_clean.py   # 可视化工具
│   ├── README.md            # 示例目录说明
│   └── COMPARE_PT_ONNX.md   # 格式对比详细文档
├── src/                   # 源代码
│   ├── models/            # 模型定义
│   │   ├── base.py       # 基类和接口
│   │   ├── ultralytics_wrapper.py # Ultralytics 包装器
│   │   └── faster_rcnn.py # Faster R-CNN 实现
│   ├── data/             # 数据集处理
│   │   └── coco_dataset.py
│   ├── metrics/          # 性能指标
│   │   └── coco_metrics.py
│   └── utils/           # 工具函数
│       ├── logger.py     # 日志和配置
│       ├── visualization.py # 可视化工具
│       └── cli.py       # CLI 接口
├── outputs/               # 输出目录
│   ├── results/          # 测试结果（JSON, CSV）
│   ├── logs/             # 运行日志
│   ├── figures/          # 可视化图表
│   └── visualizations/    # 检测框可视化
└── models_cache/          # 模型权重缓存
    ├── yolov8n.pt
    ├── yolov8s.pt
    ├── yolov8m.pt
    ├── yolov8l.pt
    └── yolov8x.pt
```

## 🎯 支持的模型

### YOLO 系列

| 模型 | 参数量 | 下载大小 | mAP@0.50:0.95 | FPS (640) |
|------|--------|----------|----------------|-----------|
| YOLOv8n | 3.16M | 6.2MB | 37.3% | 80.4 |
| YOLOv8s | 11.17M | 22MB | 44.7% | 45.7 |
| YOLOv8m | 25.90M | 50MB | 50.5% | 28.6 |
| YOLOv8l | 43.69M | 84MB | 52.9% | 20.4 |
| YOLOv8x | 68.23M | 131MB | 54.0% | 12.4 |

### 其他模型

- YOLOv9
- YOLOv10
- RT-DETR
- Faster R-CNN

## 📊 核心指标

### 评估指标

- **mAP@0.50**: IoU 阈值为 0.50 时的平均精度
- **mAP@0.50:0.95**: COCO 主指标，IoU 阈值 0.50-0.95 的平均 mAP
- **AP_small/medium/large**: 不同目标尺寸的 AP
- **AR1/AR10/AR100**: 不同检测数量的平均召回率

### 性能指标

- **FPS**: 每秒帧数（推理速度）
- **平均推理时间**: 单张图片的平均推理时间
- **模型大小**: 模型文件大小
- **参数量**: 模型参数数量

## ⚙️ 配置说明

### 数据集配置

```yaml
dataset:
  path: ~/raw/COCO          # 数据集路径
  split: val2017              # 数据集分割
  annotations: annotations/instances_val2017.json
```

### 模型配置

```yaml
models:
  - name: yolov8n
    framework: ultralytics
    weights: yolov8n.pt
    url: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### 评估配置

```yaml
evaluation:
  conf_threshold: 0.001       # 置信度阈值（推荐用于mAP评估）
  iou_threshold: 0.6          # IoU 阈值
  image_size: 640             # 输入图像尺寸
  batch_size: 32              # 批处理大小
  num_workers: 4              # 数据加载进程数
```

**重要说明**：
- `conf_threshold=0.001`: 用于完整的 mAP 评估（推荐）
- `conf_threshold=0.25`: 用于生成清晰的可视化结果
- 评估时应使用 `--conf-threshold 0.001` 参数

## 🖥️ 平台支持

### Apple Silicon (MPS)

```bash
# 方式 1: 使用启动脚本（推荐）
./run_benchmark.sh --all

# 方式 2: 手动设置环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1
python benchmark.py --all
```

**注意事项**：
- PyTorch 的 `torchvision::nms` 操作当前不支持 MPS 后端
- 必须设置 `PYTORCH_ENABLE_MPS_FALLBACK=1` 启用 CPU 回退
- 这会略微降低性能，但能正常运行

### NVIDIA CUDA

```bash
# 直接运行
python benchmark.py --all
```

### CPU

```bash
# 直接运行
python benchmark.py --all
```

## 📖 使用示例

> 💡 **提示**: 查看更多详细示例，请访问 [examples/](examples/) 目录，包含快速开始指南和所有核心功能的完整示例。

### 基础使用

```bash
# 快速测试（少量图片）
od-benchmark benchmark --model yolov8n --num-images 10

# 完整测试（所有模型）
od-benchmark benchmark --all --conf-threshold 0.001

# 测试指定模型
od-benchmark benchmark --model yolov8n --model yolov8s

# 生成可视化
od-benchmark benchmark --model yolov8n --visualize --num-viz-images 20
```

### 更多示例

```bash
# 查看所有示例
python examples/quick_start.py

# 运行基准测试示例
python examples/benchmark_example.py

# 运行模型对比示例
python examples/analyze_example.py

# 运行模型导出示例
python examples/export_example.py

# 运行格式对比示例
python examples/compare_example.py
```

### 生成可视化

```bash
# 生成检测框可视化（前10张图片）
python benchmark.py --model yolov8n --visualize

# 生成更多可视化图片
python benchmark.py --model yolov8n --visualize --num-viz-images 50

# 使用高阈值生成清晰的可视化
python benchmark.py --model yolov8n --visualize --conf-threshold 0.25
```

### 性能对比

```bash
# 对比不同配置
for model in yolov8n yolov8s yolov8m; do
  python benchmark.py --model $model --output-dir outputs/$model
done

# 对比不同阈值
for conf in 0.001 0.01 0.05 0.1 0.25; do
  python benchmark.py --model yolov8n --conf-threshold $conf
done
```

## 📝 配置文件

### config.yaml (完整评估)

```yaml
dataset:
  path: ~/raw/COCO
  split: val2017
  annotations: annotations/instances_val2017.json

models:
  - name: yolov8n
    framework: ultralytics
    weights: yolov8n.pt
    url: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
  - name: yolov8s
    framework: ultralytics
    weights: yolov8s.pt
    url: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
  # ... 更多模型

evaluation:
  conf_threshold: 0.001       # 推荐：用于完整 mAP 评估
  iou_threshold: 0.6
  batch_size: 32
  num_workers: 4
  device: auto
  image_size: 640

output:
  dir: outputs/results
  save_predictions: true
  save_visualizations: false

logging:
  level: INFO
  save_logs: true
  log_dir: outputs/logs
```

### config_test.yaml (快速测试)

```yaml
dataset:
  path: ~/raw/COCO
  split: val2017
  annotations: annotations/instances_val2017.json

models:
  - name: yolov8n
    framework: ultralytics
    weights: yolov8n.pt
    url: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

test:
  max_images: 100            # 只处理100张图片

evaluation:
  conf_threshold: 0.25        # 使用较高阈值，结果更清晰
```

## 🎨 输出说明

### 结果文件

```
outputs/
├── results/
│   ├── yolov8n_result.json     # 单个模型的详细结果
│   ├── yolov8s_result.json
│   ├── comparison.json         # 所有模型的对比结果
│   └── results_table.csv       # 性能对比表格
├── logs/
│   └── benchmark.log          # 运行日志
├── figures/
│   ├── metrics_comparison.png # 指标对比图
│   ├── fps_vs_map.png        # FPS vs mAP 图
│   └── size_vs_performance.png # 模型大小 vs 性能图
└── visualizations/
    ├── yolov8n_vis_0000_xxx.jpg  # 检测框可视化
    ├── yolov8n_vis_0001_xxx.jpg
    └── ...
```

### 结果表格 (CSV)

| Model | AP@0.50 | AP@0.50:0.95 | FPS | params |
|-------|----------|---------------|-----|--------|
| yolov8n | 0.525 | 0.373 | 80.4 | 3.16M |
| yolov8s | 0.617 | 0.447 | 45.7 | 11.17M |
| yolov8m | 0.665 | 0.505 | 28.6 | 25.90M |

## 🔧 常见问题

### 1. 添加自定义模型

**问题**: 如何添加我自己的目标检测模型？

**解决**: 查看 [添加自定义模型指南](docs/ADD_CUSTOM_MODEL.md)

**快速方式** (使用 Ultralytics 模型):
```yaml
# 在 config.yaml 中添加
models:
  - name: my_custom_yolo
    framework: ultralytics
    weights: my_custom_yolo.pt
    url: https://github.com/user/repo/releases/download/v1.0/my_custom_yolo.pt
```

```bash
# 运行
od-benchmark benchmark --model my_custom_yolo
```

**完整方式** (创建自定义模型类):
1. 创建模型类，继承 `BaseModel`
2. 在 `src/models/__init__.py` 中注册
3. 在 `config.yaml` 中配置
4. 运行测试

详细步骤请参考 [docs/ADD_CUSTOM_MODEL.md](docs/ADD_CUSTOM_MODEL.md)

### 2. MPS 后端错误

**问题**: `NotImplementedError: The operator 'torchvision::nms' is not currently implemented for the MPS device`

**解决**: 设置环境变量
```bash
 export PYTORCH_ENABLE_MPS_FALLBACK=1
 python benchmark.py --all
``` 

### 2. 模型对比分析

**问题**: 如何将我的自定义模型与基准模型进行对比分析？

**解决**: 使用 `od-benchmark analyze` 命令

快速开始：

```bash
# 快速测试（模拟模式）
python scripts/test_analysis.py

# 对比两个标准模型
od-benchmark analyze \
  --baseline yolov8n \
  --user-model yolov8s \
  --num-images 100 \
  --format all
```

详细使用指南: [模型对比分析指南](docs/ANALYSIS_USAGE.md)

### 3. mAP 指标偏低

**问题**: mAP 只有 7-10%，远低于官方的 40-50%

**原因**: 使用了过高的 `conf_threshold`（如 0.25）

**解决**: 使用低阈值进行完整评估
```bash
python benchmark.py --all --conf-threshold 0.001
```

**验证**:
- 检查日志中显示的 `AR@0.50:0.95` 应该在 35-40%
- 检查 `mAP@0.50:0.95` 应该在 40-50%

### 3. 模型权重下载失败

**问题**: 模型下载超时或失败

**解决**: 手动下载
```bash
cd models_cache
curl -L -o yolov8m.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
```

### 4. 内存不足

**问题**: 推理时内存溢出

**解决**:
1. 减小 `batch_size`
2. 使用较小的模型
3. 降低 `image_size`

## 📚 示例代码

### 可视化工具

`examples/visualize_clean.py` 是一个用于可视化目标检测模型推理结果的工具，支持：

- 单模型推理可视化
- 多模型对比
- 从 config.yaml 读取默认模型列表
- 自动检测可用模型
- 批量处理多张图片
- 生成性能对比统计
- 自动生成多模型对比缩略图

#### 快速参考

```bash
# 默认使用 config.yaml 中的所有模型
python examples/visualize_clean.py

# 只用 yolov8n
python examples/visualize_clean.py --model yolov8n

# 对比 yolov8n 和 yolov8s
python examples/visualize_clean.py --model yolov8n yolov8s

# 使用所有模型
python examples/visualize_clean.py --all
```

#### 基本用法

```bash
# 使用默认模型（从 config.yaml 读取）
python examples/visualize_clean.py

# 使用单个模型
python examples/visualize_clean.py --model yolov8n

# 使用多个模型
python examples/visualize_clean.py --model yolov8n yolov8s faster_rcnn

# 使用所有可用模型
python examples/visualize_clean.py --all
```

#### 高级选项

```bash
# 设置置信度阈值
python examples/visualize_clean.py --model yolov8n --conf-threshold 0.1

# 指定输出目录
python examples/visualize_clean.py --model yolov8n --output-dir outputs/my_vis

# 限制处理的图片数量
python examples/visualize_clean.py --model yolov8n --max-images 5

# 组合使用
python examples/visualize_clean.py \
  --model yolov8n yolov8s \
  --conf-threshold 0.2 \
  --max-images 10 \
  --output-dir outputs/comparison
```

#### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|----------|
| `--model` | 指定要使用的模型（可多个） | yolov8n, faster_rcnn |
| `--all` | 使用所有配置的模型 | False |
| `--conf-threshold` | 置信度阈值 | 0.25 |
| `--output-dir` | 输出目录 | outputs/visualizations |
| `--max-images` | 最多处理图片数量 | 全部 |
| `--config` | 指定配置文件路径 | config.yaml |

#### 输出结构

```
outputs/visualizations/
├── yolov8n/
│   ├── detection_00_000000139077.jpg
│   ├── detection_01_000000139260.jpg
│   └── ...
├── faster_rcnn/
│   ├── detection_00_000000139077.jpg
│   ├── detection_01_000000139260.jpg
│   └── ...
└── comparison/
    ├── comparison_00_000000139077.jpg
    ├── comparison_01_000000139260.jpg
    └── ...
```

每个模型都有自己的子目录，便于对比不同模型的检测结果。

#### 对比缩略图

当使用多个模型时，会自动生成对比缩略图，方便直观比较不同模型的检测效果：

- 2 个模型：水平并排展示
- 3 个模型：水平并排展示
- 4 个以上模型：2×2 或更大网格布局

对比图保存在 `outputs/visualizations/comparison/` 目录中。

#### 支持的模型

##### YOLO 系列
```
YOLOv8: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
YOLOv9: yolov9t, yolov9s, yolov9m
YOLOv10: yolov10n, yolov10s, yolov10m, yolov10b
```

##### 其他模型
```
faster_rcnn (使用 torchvision 内置预训练权重)
```

#### 性能对比

当使用多个模型时，会自动生成对比表格：

```
================================================================================
[对比] 模型性能对比
================================================================================
模型              成功         总检测        平均/图      
--------------------------------------------------
faster_rcnn     2          50         25.00     
rtdetr-l        1          57         57.00     
yolov8n         2          9          4.50     
================================================================================
```

#### 使用示例

##### 示例 1：快速测试

```bash
# 只处理第一张图片，快速测试 yolov8n
python examples/visualize_clean.py --model yolov8n --max-images 1
```

##### 示例 2：对比不同大小的 YOLO 模型

```bash
# 对比 yolov8n, yolov8s, yolov8m
python examples/visualize_clean.py \
  --model yolov8n yolov8s yolov8m \
  --max-images 5
```

##### 示例 3：对比 YOLO 和 Faster R-CNN

```bash
# 对比单阶段和两阶段检测器
python examples/visualize_clean.py \
  --model yolov8n yolov8s faster_rcnn \
  --conf-threshold 0.1
```

##### 示例 4：完整对比

```bash
# 使用所有可用模型，低置信度阈值
export PYTORCH_ENABLE_MPS_FALLBACK=1
python examples/visualize_clean.py \
  --all \
  --conf-threshold 0.05 \
  --max-images 10
```

#### 常用命令

```bash
# 对比不同大小的 YOLO 模型
python examples/visualize_clean.py \
  --model yolov8n yolov8s yolov8m yolov8l \
  --max-images 10

# 对比 YOLOv8 和 Faster R-CNN
python examples/visualize_clean.py \
  --model yolov8n yolov8s faster_rcnn \
  --conf-threshold 0.1

# 快速测试（只处理 1 张图片）
python examples/visualize_clean.py \
  --model yolov8n \
  --max-images 1

# 自定义输出目录
python examples/visualize_clean.py \
  --model yolov8n yolov8s \
  --output-dir outputs/my_test
```

#### 查看结果

```bash
# 查看输出目录结构
ls -la outputs/visualizations/

# 查看特定模型的结果
ls -la outputs/visualizations/yolov8n/

# 使用图片查看器打开
open outputs/visualizations/yolov8n/
```

#### 性能对比输出

当使用多个模型时，会自动生成对比表格：

```
模型              成功         总检测        平均/图      
--------------------------------------------------
faster_rcnn     2          50         25.00     
rtdetr-l        1          57         57.00     
yolov8n         2          9          4.50     
```

#### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model` | 指定模型（可多个） | `--model yolov8n yolov8s` |
| `--all` | 使用所有配置的模型 | `--all` |
| `--conf-threshold` | 置信度阈值 | `--conf-threshold 0.1` |
| `--output-dir` | 输出目录 | `--output-dir outputs/test` |
| `--max-images` | 最多处理图片数量 | `--max-images 5` |

#### 注意事项

1. **MPS 设备支持**：在 Apple Silicon 上运行时，需要设置环境变量：
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```

2. **数据集路径**：确保 COCO 验证集路径正确：
   ```
   ~/raw/COCO/val2017/
   ```

3. **模型权重**：确保模型权重在 `models_cache/` 目录中，或使用内置预训练权重（如 faster_rcnn）

4. **输出目录**：输出图片会保存到指定目录，每个模型独立子目录

5. **检测框限制**：每张图片最多绘制 10 个检测框（可在代码中修改 `max_boxes` 参数）

## 🚧 待办事项

### 短期 (1-2周)

- [ ] 添加进度条（tqdm）
- [ ] 添加基础单元测试
- [ ] 完善类型提示
- [ ] 改进错误处理

### 中期 (1-2月)

- [ ] 添加 PR 曲线可视化
- [ ] 添加混淆矩阵
- [ ] 添加类别级别 mAP
- [ ] 性能优化（多进程、批处理）

### 长期 (3-6月)

- [ ] 添加完整的单元测试覆盖
- [ ] 生成 API 文档（Sphinx）
- [ ] 添加 CI/CD（GitHub Actions）
- [ ] 支持 ONNX/TensorRT 导出

## 🤝 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'feat: Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 LICENSE 文件了解详情

## 📞 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

---

**注意**: Apple Silicon (MPS) 用户请确保设置 `PYTORCH_ENABLE_MPS_FALLBACK=1` 环境变量以避免 `torchvision::nms` 错误。
