# 模型对比分析功能设计

## 功能概述

将 config.yaml 中配置的模型作为基准模型，导入用户自定义的模型作为待分析模型，运行对比测试并输出分析报告。

## 设计方案

### 命令结构

```bash
od-benchmark analyze \
  --baseline <baseline_model> \
  --user-model <user_model_config> \
  --num-images <count> \
  --output <output_dir>
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|---------|
| `--baseline` | 基准模型（从 config.yaml 中选择） | - |
| `--user-model` | 用户模型（可以是：模型名、权重文件路径、或配置路径） | - |
| `--num-images` | 测试图片数量 | 50 |
| `--config` | 配置文件路径 | config.yaml |
| `--output-dir` | 输出目录 | outputs/analysis |
| `--metrics` | 对比指标（mAP、FPS、延迟等） | all |
| `--format` | 输出格式（json, html, csv） | html |

## 用户模型支持格式

### 格式 1: 使用已有模型名
```bash
od-benchmark analyze \
  --baseline yolov8n \
  --user-model yolov8s \
  --num-images 100
```

### 格式 2: 指定权重文件
```bash
od-benchmark analyze \
  --baseline yolov8n \
  --user-model path/to/my_model.pt \
  --num-images 100
```

### 格式 3: 使用配置文件
```bash
od-benchmark analyze \
  --baseline yolov8n \
  --user-model user_models/my_custom_model.yaml \
  --num-images 100
```

## 输出内容

### 1. 对比指标

- mAP@0.50
- mAP@0.50:0.95
- AP@0.50:0.95 (small/medium/large)
- FPS (每秒帧数)
- 平均推理时间
- 模型大小
- 参数量

### 2. 性能对比

- 相对性能提升 (%)
- 绝对性能差异
- 速度对比
- 精度对比

### 3. 可视化

- 指标对比柱状图
- 性能雷达图
- 推理时间分布图

### 4. 详细报告

- 每个指标的详细对比
- 优劣势分析
- 推荐使用场景

## 输出格式

### JSON
```json
{
  "baseline": {
    "name": "yolov8n",
    "metrics": {...}
  },
  "user_model": {
    "name": "custom_model",
    "metrics": {...}
  },
  "comparison": {
    "map_diff": 5.2,
    "fps_diff": -15.3,
    "speedup": 0.85
  },
  "recommendation": "..."
}
```

### HTML
- 交互式报告
- 可视化图表
- 表格对比

### CSV
- 便于数据分析
- 适合导入 Excel

## 实现计划

### Phase 1: 核心功能
- [ ] 添加 `analyze` 子命令到 src/cli.py
- [ ] 实现模型对比函数
- [ ] 实现指标计算
- [ ] 实现 JSON 输出

### Phase 2: 可视化
- [ ] 添加图表生成
- [ ] 实现 HTML 报告
- [ ] 添加 CSV 导出

### Phase 3: 高级功能
- [ ] 多模型批量对比
- [ ] 自定义指标
- [ ] 热力图可视化

## 使用示例

### 示例 1: 对比 YOLOv8n 和 YOLOv8s
```bash
od-benchmark analyze \
  --baseline yolov8n \
  --user-model yolov8s \
  --num-images 200 \
  --output-dir outputs/analysis/001
```

### 示例 2: 对比基准和自定义模型
```bash
od-benchmark analyze \
  --baseline yolov8n \
  --user-model path/to/custom_model.pt \
  --num-images 100 \
  --format html
```

### 示例 3: 模拟模式（开发调试用）
```bash
# 使用 yolov8n 模拟用户自定义模型
# 便于在开发过程中测试和验证
od-benchmark analyze \
  --baseline yolov8n \
  --user-model yolov8n:simulated \
  --num-images 10 \
  --debug
```

## 配置文件格式

### user_models/my_model.yaml
```yaml
name: "My Custom Model"
description: "Custom object detection model"
framework: custom
model_type: yolov8  # 或其他支持的类型
weights: "path/to/weights.pt"
input_size: 640
class_names:
  0: person
  1: car
  # ...
```

## 调试和演示

### 模拟模式
在开发过程中，可以使用 `--simulated` 模式快速测试：

```bash
od-benchmark analyze \
  --baseline yolov8n \
  --user-model yolov8n:simulated \
  --num-images 10
```

这样：
- 不需要下载额外的模型
- 可以快速验证对比流程
- 便于调试可视化代码
