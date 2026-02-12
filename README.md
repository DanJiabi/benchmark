# Benchmark - Object Detection Performance Evaluation

基于 COCO 数据集的目标检测模型性能基准测试项目。

## 环境设置

```bash
# 创建虚拟环境
mamba env create -f environment.yml --force

# 激活环境
conda activate benchmark
```

## 快速开始

```bash
# 激活环境
conda activate benchmark

# 运行所有模型基准测试（两种方式）
python benchmark.py --config config.yaml --all
python benchmark.py --config config.yaml --model all

# 运行指定模型
python benchmark.py --config config.yaml --model yolov8n --model yolov8s

# 运行测试（仅处理少量图片）
python benchmark.py --config config_test.yaml --model yolov8n
```

### 模型权重

模型权重会自动从 GitHub 下载并缓存到 `models_cache/` 目录。

已下载的模型：
- ✅ yolov8n.pt (6.2MB, 3.16M 参数)
- ✅ yolov8s.pt (22MB, 11.17M 参数)
- ⚠️  yolov8m.pt (下载可能超时，需手动下载)
- ✅ yolov8l.pt (23MB, 43.67M 参数)
- ✅ yolov8x.pt (8.3MB, 68.17M 参数)

手动下载 yolov8m.pt:
```bash
curl -L -o models_cache/yolov8m.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
```

### Apple Silicon (MPS) 用户

```bash
# 方式 1：使用启动脚本（推荐）
./run_benchmark.sh --config config.yaml --model yolov8n

# 方式 2：手动设置环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1
python benchmark.py --config config.yaml --model yolov8n
```

### NVIDIA CUDA 用户

```bash
# 直接运行
python benchmark.py --config config.yaml --all
```

### CPU 用户

```bash
# 直接运行
python benchmark.py --config config.yaml --model yolov8n
```

### 常用命令

```bash
# 运行所有模型基准测试（两种方式）
python benchmark.py --config config.yaml --all
python benchmark.py --config config.yaml --model all

# 运行指定模型
python benchmark.py --config config.yaml --model yolov8n --model yolov8s

# 运行测试（仅处理少量图片）
python benchmark.py --config config_test.yaml --model yolov8n
```

### 注意事项

1. **Apple Silicon MPS 设备**：
   - PyTorch 的 torchvision::nms 操作当前不支持 MPS 后端
   - 必须设置 `PYTORCH_ENABLE_MPS_FALLBACK=1` 启用 CPU 回退
   - 或使用提供的 `run_benchmark.sh` 启动脚本自动设置
   - 这会略微降低性能，但能正常运行

2. **数据集处理时间**：
   - 完整的 val2017 数据集（5000 张图片）需要较长时间
   - 建议先用 `config_test.yaml` 测试少量图片

## 项目结构

```
benchmark/
├── environment.yml          # Conda 环境配置
├── config.yaml              # 模型和测试配置
├── requirements.txt         # Python 依赖
├── src/
│   ├── models/              # 模型加载器
│   ├── metrics/             # 指标计算
│   ├── data/                # 数据处理
│   ├── benchmark.py         # 主运行脚本
│   └── utils/               # 工具函数
├── outputs/
│   ├── results/             # 测试结果
│   ├── logs/                # 运行日志
│   └── figures/             # 可视化图表
└── models_cache/            # 模型权重缓存
```

## 支持的模型

- YOLOv8 (n/s/m/l/x)
- YOLOv9
- YOLOv10
- YOLOv11
- RT-DETR
- Faster R-CNN

## 核心指标

- mAP@0.5
- mAP@0.5:0.95 (COCO 主指标)
- AP50/AP75
- AP_small/medium/large
- AR1/AR10/AR100
- FPS / 延迟
- 模型大小 / FLOPs

## 数据集

使用 COCO 2017 数据集，位于 `~/raw/COCO`
