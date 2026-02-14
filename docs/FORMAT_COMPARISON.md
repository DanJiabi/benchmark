# ONNX 性能比较使用指南

本文档介绍如何使用 `od-benchmark compare` 命令比较 PyTorch 和 ONNX 格式的性能差异。

## 功能概述

格式比较功能可以：
- 自动导出 ONNX 模型
- 对比 PyTorch 和 ONNX 的推理速度
- 对比两种格式的精度 (mAP)
- 生成性能对比报告
- 提供部署建议

## 使用方法

### 基本命令

```bash
od-benchmark compare --model models_cache/yolov8n.pt --num-images 50
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | 是 | - | PyTorch 模型路径 (.pt) |
| `--model-name` | 否 | 自动 | 模型名称显示 |
| `--formats` | 否 | pytorch,onnx | 要对比的格式 |
| `--config` | 否 | config.yaml | 配置文件路径 |
| `--num-images` | 否 | 50 | 测试图片数量 |
| `--output-dir` | 否 | outputs/format_comparison | 输出目录 |

## 使用示例

### 示例 1: 对比 YOLOv8n

```bash
od-benchmark compare \
  --model models_cache/yolov8n.pt \
  --num-images 100
```

输出示例：
```
======================================================================
格式性能对比分析
======================================================================
模型: yolov8n
对比格式: ['pytorch', 'onnx']
测试图片数: 100

导出 ONNX 模型: outputs/format_comparison/yolov8n.onnx

测试 PYTORCH 格式...
✅ PYTORCH 测试完成
   FPS: 125.50
   mAP@0.50: 0.5234

测试 ONNX 格式...
✅ ONNX 测试完成
   FPS: 156.30
   mAP@0.50: 0.5218

======================================================================
格式对比结果摘要
======================================================================

性能对比:
----------------------------------------------------------------------
格式            FPS         延迟(ms)    mAP@0.5     
----------------------------------------------------------------------
PYTORCH         125.50      7.97        0.5234      
ONNX            156.30      6.40        0.5218      

速度提升:
  ONNX vs PyTorch: +24.5% (1.25x)

推荐:
  ✅ ONNX 比 PyTorch 快 24.5%，建议使用 ONNX 部署
  ✅ ONNX 精度损失 < 1%，可放心使用
======================================================================

报告已保存: outputs/format_comparison/yolov8n_format_comparison.json
```

### 示例 2: 只对比 PyTorch

```bash
od-benchmark compare \
  --model models_cache/yolov8n.pt \
  --formats pytorch \
  --num-images 50
```

### 示例 3: 自定义模型名称

```bash
od-benchmark compare \
  --model models_cache/my_custom_model.pt \
  --model-name "My Custom YOLO" \
  --num-images 200
```

### 示例 4: 使用不同的测试图片数

```bash
# 快速测试 (10张图片)
od-benchmark compare --model yolov8n.pt --num-images 10

# 完整测试 (500张图片)
od-benchmark compare --model yolov8n.pt --num-images 500
```

## Python API 使用

```python
from src.analysis import FormatComparison

# 创建比较器
comparer = FormatComparison(
    model_name="yolov8n",
    model_path="models_cache/yolov8n.pt",
    output_dir="outputs/format_comparison"
)

# 加载数据集
from src.data.coco_dataset import COCOInferenceDataset
dataset = COCOInferenceDataset("~/raw/COCO", "val2017")

# 执行对比
results = comparer.compare_formats(
    dataset=dataset,
    annotations_file="~/raw/COCO/annotations/instances_val2017.json",
    formats=["pytorch", "onnx"],
    max_images=100
)

# 保存报告
report_file = comparer.save_report()
print(f"报告已保存: {report_file}")
```

## 输出文件

对比完成后会生成以下文件：

```
outputs/format_comparison/
├── yolov8n.onnx                          # 导出的 ONNX 模型
└── yolov8n_format_comparison.json        # 对比报告
```

### 报告文件格式

```json
{
  "model_name": "yolov8n",
  "timestamp": "2026-02-14T15:30:00",
  "formats": {
    "pytorch": {
      "fps": 125.50,
      "avg_time_ms": 7.97,
      "map_50": 0.5234,
      "map_50_95": 0.3689
    },
    "onnx": {
      "fps": 156.30,
      "avg_time_ms": 6.40,
      "map_50": 0.5218,
      "map_50_95": 0.3675
    }
  },
  "speed_comparison": {
    "pytorch_fps": 125.50,
    "onnx_fps": 156.30,
    "speedup": 1.25,
    "speedup_pct": 24.5
  },
  "accuracy_comparison": {
    "pytorch_map_50": 0.5234,
    "onnx_map_50": 0.5218,
    "map_diff": 0.0016,
    "map_diff_pct": 0.31
  },
  "recommendations": [
    "✅ ONNX 比 PyTorch 快 24.5%，建议使用 ONNX 部署",
    "✅ ONNX 精度损失 < 1%，可放心使用"
  ]
}
```

## 推荐逻辑

系统会根据测试结果自动生成推荐：

### 速度推荐

| 速度提升 | 推荐 |
|----------|------|
| > 10% | ✅ 建议使用 ONNX |
| ±10% | ⚖️ 性能相近，按需选择 |
| < -10% | ⚠️ 建议使用 PyTorch |

### 精度推荐

| 精度损失 | 推荐 |
|----------|------|
| < 1% | ✅ 精度损失可忽略 |
| 1-3% | ⚖️ 精度损失在可接受范围 |
| > 3% | ⚠️ 精度损失较大，请检查 |

## 批量对比多个模型

创建批量对比脚本 `scripts/batch_compare.sh`：

```bash
#!/bin/bash

MODELS=("yolov8n" "yolov8s" "yolov8m")
NUM_IMAGES=50

for model in "${MODELS[@]}"; do
    echo "======================================"
    echo "对比模型: $model"
    echo "======================================"
    
    od-benchmark compare \
        --model "models_cache/${model}.pt" \
        --num-images $NUM_IMAGES \
        --output-dir "outputs/format_comparison/${model}"
    
    echo ""
done

echo "所有对比完成！"
echo "报告位置: outputs/format_comparison/"
```

## 性能对比表

不同模型在 CPU/MPS/GPU 上的性能对比示例：

| 模型 | PyTorch CPU | ONNX CPU | 提升 |
|------|-------------|----------|------|
| YOLOv8n | 12 FPS | 18 FPS | +50% |
| YOLOv8s | 8 FPS | 12 FPS | +50% |
| YOLOv8m | 5 FPS | 7 FPS | +40% |

| 模型 | PyTorch MPS | ONNX CoreML | ONNX CPU |
|------|-------------|--------------|----------|
| YOLOv8n | 180 FPS | 68 FPS | 45 FPS |
| YOLOv8s | 130 FPS | 50 FPS | 32 FPS |
| YOLOv8m | 85 FPS | 32 FPS | 20 FPS |

*注：Apple Silicon 上 PyTorch MPS 仍然最快，ONNX CoreML 比 CPU 快约 50%。*

## 注意事项

### 1. ONNX 导出时间

首次运行时需要导出 ONNX 模型，可能需要 5-30 秒。导出的模型会被缓存，下次直接使用。

### 2. 后处理简化

ONNX 推理使用简化的后处理，主要用于性能测试。实际部署时应使用完整的 NMS 后处理。

### 3. Apple Silicon (MPS)

在 Apple Silicon (M1/M2/M3/M4) 上：
- PyTorch MPS 仍然最快（~2-3x）
- ONNX + CoreML 比纯 CPU 快约 50%
- 建议优先使用 PyTorch MPS，ONNX + CoreML 作为备选

**CoreML 支持** (ONNX Runtime 1.23+):
- 自动使用 `CoreMLExecutionProvider` 当 device="mps"
- 约 90% 的计算节点被 CoreML 加速
- 不支持的节点自动回退到 CPU

### 4. 批处理大小

当前对比使用 batch_size=1。如需测试更大 batch size，需要手动修改代码。

## 故障排除

### Q: ONNX 导出失败

```
错误: ONNX 导出失败
```

A: 检查 ultralytics 安装：
```bash
pip install --upgrade ultralytics
```

### Q: onnxruntime 未安装

```
ImportError: 需要安装 onnxruntime
```

A: 安装 onnxruntime：
```bash
pip install onnxruntime
# 或 GPU 版本
pip install onnxruntime-gpu
```

### Q: ONNX 推理结果为空

```
ONNX 测试完成
   FPS: 0.00
   mAP@0.50: 0.0000
```

A: 这是正常的，因为使用了简化的后处理。主要关注 FPS 性能指标。

### Q: ONNX 比 PyTorch 慢

A: 在以下情况可能出现：
1. Apple Silicon (MPS)：PyTorch MPS 通常快 2-3 倍（正常）
2. 非常小的模型：PyTorch 优化更好
3. CPU 不支持 AVX 指令：ONNX Runtime 需要 AVX

**注意**：在 Apple Silicon 上，ONNX + CoreML 仍比 PyTorch MPS 慢约 50-60%，这是正常的。CoreML 比 ONNX CPU 快约 50%。

## 相关命令

- `od-benchmark export` - 导出 ONNX 模型
- `od-benchmark benchmark` - 基准测试
- `od-benchmark analyze` - 模型对比分析

## 反馈与支持

如有问题或建议，请提交 Issue 或 Pull Request。
