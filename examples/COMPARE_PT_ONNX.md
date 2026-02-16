# PyTorch vs ONNX 推理对比工具

## 功能概述

`compare_pt_onnx.py` 脚本用于对比 PyTorch 和 ONNX 版本的同一个模型的推理结果，帮助您验证 ONNX 导出和推理是否正确。

## 快速开始

### 基本用法

```bash
# 对比 yolov10n 的 PyTorch 和 ONNX 版本
python examples/compare_pt_onnx.py \
  --pt-model yolov10n \
  --onnx-model models_export/yolov10n.onnx \
  --device cpu \
  --max-images 5
```

### 使用说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--pt-model` | 是 | PyTorch 模型名称或路径 | - |
| `--onnx-model` | 是 | ONNX 模型文件路径 | - |
| `--device` | 否 | 推理设备 | cpu |
| `--conf-threshold` | 否 | 置信度阈值 | 0.25 |
| `--output-dir` | 否 | 输出目录 | outputs/onnx_comparison |
| `--max-images` | 否 | 最多处理图片数量 | 全部 |

## 使用示例

### 示例 1: 对比 YOLOv8n

```bash
python examples/compare_pt_onnx.py \
  --pt-model yolov8n \
  --onnx-model models_export/yolov8n.onnx \
  --device cpu \
  --max-images 3
```

### 示例 2: 使用特定模型权重文件

```bash
# 假设您有自定义权重
python examples/compare_pt_onnx.py \
  --pt-model models_cache/my_custom_model.pt \
  --onnx-model models_export/my_custom_model.onnx \
  --device mps
```

### 示例 3: 不同置信度阈值测试

```bash
# 低阈值（检测更多）
python examples/compare_pt_onnx.py \
  --pt-model yolov8n \
  --onnx-model models_export/yolov8n.onnx \
  --conf-threshold 0.1

# 高阈值（检测更少但更准确）
python examples/compare_pt_onnx.py \
  --pt-model yolov8n \
  --onnx-model models_export/yolov8n.onnx \
  --conf-threshold 0.5
```

## 输出说明

### 1. 对比图片 (`outputs/onnx_comparison/{model_name}/`)

每个对比图包含：
- **左侧**: PyTorch 模型检测结果
- **右侧**: ONNX 模型检测结果
- **顶部**: 统计信息（检测数量、相似度）

文件命名：`comparison_{index}_{image_name}.jpg`

### 2. 汇总报告 (`summary.txt`)

包含：
- 总图片数和成功处理数
- PyTorch 平均检测数
- ONNX 平均检测数
- 检测数量差异
- 平均相似度（百分比）
- 每张图片的详细结果

### 3. 相似度计算

相似度基于检测框数量计算：
```
相似度 = 1 - |PT数量 - ONNX数量| / max(PT数量, ONNX数量)
```

- **100%**: 检测数量完全相同
- **0%**: 差异很大
- **注意**: 这只是一个粗略的相似度指标，实际的精度对比需要更详细的 IoU 计算

## 结果解读

### 情况 1: 高相似度 (>80%)

**说明**: ONNX 推理结果与 PyTorch 非常接近

**可能原因**：
- ONNX 导出成功
- ONNX 后处理正确

**建议**: ✅ 可以放心使用 ONNX 版本

### 情况 2: 中等相似度 (50-80%)

**说明**: ONNX 推理结果与 PyTorch 有一定差异

**可能原因**：
- ONNX 使用简化的后处理
- 置信度阈值敏感度不同

**建议**: ⚖️ 可以使用 ONNX 版本，但需要进一步验证精度

### 情况 3: 低相似度 (<50%)

**说明**: ONNX 推理结果与 PyTorch 差异很大

**可能原因**：
- ONNX 后处理有问题
- ONNX 模型导出时精度丢失

**建议**: ❌ 需要修复 ONNX 导出或后处理

### 常见问题

#### Q: ONNX 检测框数量远多于 PyTorch

**示例**: PyTorch 检测 5 个，ONNX 检测 20 个

**原因**: 
- ONNX 后处理使用简化版本，没有正确的 NMS（非极大值抑制）
- 当前实现只返回前 10 个预测，没有过滤低置信度框

**解决**: 
1. 检查 ONNX 输出格式
2. 实现完整的后处理逻辑（NMS）
3. 使用 Ultralytics 的后处理或 onnxruntime 的 NMS

#### Q: ONNX 检测框位置不准确

**原因**: 
- 坐标系统不一致（xyxy vs xywh）
- 缩放因子错误

**解决**: 
1. 检查 ONNX 模型的输出格式
2. 调整后处理的坐标转换逻辑

#### Q: ONNX 推理速度比 PyTorch 慢

**可能原因**：
- 使用 CPU 而不是 GPU/MPS
- ONNX 模型没有优化

**解决**：
```bash
# 检查 ONNX Runtime 使用的执行提供程序
# 使用 --device 指定正确的设备
python examples/compare_pt_onnx.py \
  --pt-model yolov8n \
  --onnx-model models_export/yolov8n.onnx \
  --device mps  # 使用 MPS (Apple Silicon)
```

## 注意事项

### ONNX 后处理

**重要**: 当前 `ONNXModel` 使用**简化的后处理**，主要用于性能测试。实际生产环境应该使用完整的后处理逻辑，包括：
- 正确的 NMS（非极大值抑制）
- 类别置信度提取
- 坐标格式转换

完整的后处理参考：
```python
# 应该使用类似 Ultralytics 的后处理
from ultralytics.utils import ops

# NMS
boxes = ops.non_max_suppression(
    predictions[0],
    conf_thres=conf_threshold,
    iou_thres=0.45,
    max_det=300
)
```

### 设备选择

| 设备 | 说明 | 推荐场景 |
|------|------|----------|
| `cpu` | 使用 CPU 推理 | 测试 ONNX Runtime 安装 |
| `mps` | 使用 MPS (Apple Silicon) | Apple Silicon 设备 |
| `cuda` | 使用 CUDA (NVIDIA GPU) | NVIDIA GPU 设备 |

### 置信度阈值

不同的阈值会影响检测结果：
- **0.001**: 适合完整 mAP 评估（检测很多）
- **0.25**: 适合可视化（检测适中）
- **0.5**: 高置信度（检测较少但准确）

## 相关文档

- [ONNX 导出指南](docs/EXPORT_GUIDE.md)
- [格式对比指南](docs/FORMAT_COMPARISON.md)
- [添加自定义模型](docs/ADD_CUSTOM_MODEL.md)

## 故障排除

### Q: 导入错误 `ImportError: No module named 'onnxruntime'`

**A**: 安装 onnxruntime
```bash
pip install onnxruntime

# 或者 GPU 版本
pip install onnxruntime-gpu
```

### Q: ONNX 模型文件不存在

**A**: 确保已导出 ONNX 模型
```bash
# 检查导出的模型
ls models_export/*.onnx

# 如果没有，先导出
od-benchmark export --all-models --format onnx
```

### Q: 图片读取失败

**A**: 检查数据集路径
```bash
# 检查 COCO 数据集路径
ls ~/raw/COCO/val2017/*.jpg

# 如果路径不同，修改脚本中的 COCO_VAL_PATH 变量
```

## 反馈与支持

如有问题或建议，请提交 Issue 或 Pull Request。
