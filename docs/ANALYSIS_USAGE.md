# 模型对比分析使用指南

## 功能概述

`od-benchmark analyze` 命令允许您将 config.yaml 中配置的模型作为基准，与您自己的自定义模型进行对比分析，并生成详细的对比报告。

## 快速开始

### 安装并验证

```bash
# 安装包
pip install -e .

# 验证安装
python scripts/test_installation.py
```

### 快速测试（模拟模式）

运行快速测试脚本：

```bash
python scripts/test_analysis.py
```

这会运行一个模拟的对比测试，使用：
- 基准模型：yolov8n（真实模型）
- 用户模型：yolov8n:simulated（模拟的自定义模型）

## 命令使用

### 基本语法

```bash
od-benchmark analyze \
  --baseline <model_name> \
  --user-model <user_model> \
  --num-images <count>
```

### 参数详解

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|---------|
| `--baseline` | 是 | 基准模型名（从 config.yaml 选择） | - |
| `--user-model` | 是 | 用户模型（模型名、权重文件路径或配置文件） | - |
| `--config` | 否 | 配置文件路径 | config.yaml |
| `--num-images` | 否 | 测试图片数量 | 50 |
| `--output-dir` | 否 | 输出目录 | outputs/analysis |
| `--format` | 否 | 输出格式（json/html/csv/all） | all |
| `--metrics` | 否 | 指标（all/mAP/fps/latency） | all |
| `--debug` | 否 | 调试模式 | false |

## 使用示例

### 示例 1: 对比 YOLOv8n 和 YOLOv8s

```bash
od-benchmark analyze \
  --baseline yolov8n \
  --user-model yolov8s \
  --num-images 100 \
  --format all
```

### 示例 2: 使用自定义权重文件

```bash
od-benchmark analyze \
  --baseline yolov8n \
  --user-model /path/to/my_model.pt \
  --num-images 50 \
  --format html
```

### 示例 3: 使用用户模型配置文件

首先创建配置文件 `user_models/my_custom.yaml`：

```yaml
name: "My Custom Model"
description: "Custom object detection model"
framework: custom
model_type: yolov8
weights: "path/to/weights.pt"
input_size: 640

class_names:
  0: person
   1: car
  # ... 添加更多类别

parameters:
  conf_threshold: 0.25
  iou_threshold: 0.5
```

然后运行分析：

```bash
od-benchmark analyze \
  --baseline yolov8n \
  --user-model user_models/my_custom.yaml \
  --num-images 200 \
  --format all
```

### 示例 4: 使用模拟模式（开发调试）

```bash
od-benchmark analyze \
  --baseline yolov8n \
  --user-model yolov8n:simulated \
  --num-images 10 \
  --debug
```

## 输出内容

### 1. 控制台输出

分析过程会在控制台实时显示：
- 模型加载状态
- 推理进度
- 指标对比结果
- 推荐意见

### 2. JSON 报告 (`outputs/analysis/comparison.json`)

包含所有指标的详细对比数据。

### 3. HTML 报告 (`outputs/analysis/comparison.html`)

可交互式查看的详细报告，包括：
- 模型信息卡片
- 指标对比表格
- 推荐意见列表
- 响应式设计

### 4. CSV 报告 (`outputs/analysis/comparison.csv`)

适合进一步数据分析。

## 指标说明

### 准确性指标

- **mAP@0.50**: IoU 阈值 0.50 时的平均精度
- **mAP@0.50:0.95**: COCO 主指标，IoU 阈值 0.50-0.95 的平均 mAP
- **AP@small**: 小目标 AP
- **AP@medium**: 中等目标 AP
- **AP@large**: 大目标 AP

### 性能指标

- **FPS**: 每秒帧数
- **平均推理时间**: 单张图片的平均推理时间（毫秒）
- **加速比**: 用户模型速度 / 基准模型速度
- **速度差异 %**: (用户FPS - 基准FPS) / 基准FPS × 100%

### 模型信息

- **参数量**: 模型参数数量（百万）
- **模型大小**: 模型文件大小（MB）

## 推荐逻辑

### 准确性分析

| 条件 | 推荐 |
|------|------|
| 用户 mAP 高于基准 > 5% | ✅ 用户模型准确率明显更高 |
| 用户 mAP 高于基准 > 1% | ✅ 用户模型准确率有所提升 |
| 用户 mAP 低于基准 > 5% | ⚠️  用户模型准确率明显低于基准 |
| 用户 mAP 低于基准 > 1% | ⚠️ 用户模型准确率略低于基准 |

### 速度分析

| 条件 | 推荐 |
|------|------|
| 用户 FPS 快 > 50% | ✅ 用户模型速度快很多 |
| 用户 FPS 快 > 10% | ✅ 用户模型明显更快 |
| 用户 FPS 慢 > 50% | ⚠️  用户模型速度慢很多 |
| 用户 FPS 慢 > 10% | ⚠️  用户模型明显更慢 |

### 权衡分析

| 场景 | 推荐 |
|------|------|
| 用户模型准确率 + 速度都更好 | 🎉 采用用户模型 |
| 用户模型准确率更好但速度更慢 | ⚖️ 权衡：根据应用场景选择 |
| 用户模型速度更快但准确率略低 | ⚖️ 权衡：根据应用场景选择 |
| 用户模型准确率更低 | ❌ 需要改进用户模型 |

## 用户模型支持

### 支持的格式

1. **模型名称**: config.yaml 中已配置的模型
   ```bash
   od-benchmark analyze --baseline yolov8n --user-model yolov8s
   ```

2. **权重文件**: 直接指定权重文件路径
   ```bash
   od-benchmark analyze --baseline yolov8n --user-model models_cache/custom.pt
   ```

3. **配置文件**: 使用 YAML 配置文件描述模型
   ```bash
   od-benchmark analyze --baseline yolov8n --user-model user_models/my_model.yaml
   ```

4. **模拟模式**: 使用内置模拟模型（用于开发调试）
   ```bash
   od-benchmark analyze --baseline yolov8n --user-model yolov8n:simulated --debug
   ```

### 模拟模式说明

模拟模式使用 `yolov8n:simulated` 格式，它是一个包装的真实 yolov8n 模型，但会：
- 降低 15% 的检测置信度（模拟准确率较低的模型）
- 可以添加延迟（开发时）

这允许在不下载额外模型的情况下测试对比分析功能。

### 模拟模式支持的变体

- `yolov8n:simulated` - 标准模拟（降低 15% 置信度）
- `yolov8n:simulated:fast` - 快速模拟（降低 30% 置信度）
- `yolov8n:simulated:slow` - 慢速模拟（降低 5% 置信度）

## 开发指南

### 自定义用户模型

#### 方式 1: 使用已有框架（简单）

如果您的模型使用 Ultralytics YOLO 或其他已支持的框架：

1. 在 config.yaml 中添加模型配置
2. 直接使用模型名运行分析

#### 方式 2: 创建自定义模型类（高级）

1. 继承 `BaseModel`
2. 实现必需方法：
   - `load_model(weights_path)` - 加载模型权重
   - `predict(image, conf_threshold)` - 执行推理，返回 `List[Detection]`
   - `get_model_info()` - 返回模型信息
   - `warmup(image_size)` - 模型预热

3. 在 `src/models/__init__.py` 中注册

详细步骤参考：[添加自定义模型指南](docs/ADD_CUSTOM_MODEL.md)

### 调试技巧

1. **使用少量图片**: `--num-images 5`
2. **启用调试模式**: `--debug`
3. **查看日志**: 检查控制台输出
4. **检查 JSON 报告**: 查看 `outputs/analysis/comparison.json`

## 常见问题

### Q: 如何对比多个用户模型？

A: 依次运行多次分析，每次指定不同的用户模型：
```bash
od-benchmark analyze --baseline yolov8n --user-model model1 --num-images 50
od-benchmark analyze --baseline yolov8n --user-model model2 --num-images 50
od-benchmark analyze --baseline yolov8n --user-model model3 --num-images 50
```

### Q: 如何只评估性能，不使用 COCO 标注？

A: 当前版本需要 COCO 数据集和标注文件以计算 mAP。如果您只需要评估推理速度，可以修改代码跳过 COCO 指标计算。

### Q: 如何添加自定义指标？

A: 在 `src/analysis/model_comparison.py` 中的 `_generate_comparison()` 方法中添加自定义指标计算逻辑。

### Q: 模拟模式和真实模型有什么区别？

A: 模拟模式只是用于测试和演示。真实对比应该使用您的实际自定义模型。模拟模式的唯一区别是会降低检测置信度来模拟不同的准确率。

## 项目结构

```
benchmark/
├── src/
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── model_comparison.py  # 模型对比核心逻辑
│   └── models/
│       ├── simulated_model.py     # 模拟模型和用户模型加载器
│       ├── __init__.py              # 模型注册（已更新）
│       └── ...
└── scripts/
    └── test_analysis.py           # 快速测试脚本
```

## 下一步

1. 阅读 [添加自定义模型指南](docs/ADD_CUSTOM_MODEL.md) 了解如何创建自定义模型
2. 运行 `python scripts/test_analysis.py` 进行快速测试
3. 使用 `od-benchmark analyze --help` 查看所有选项
4. 开始对比您的模型！

## 反馈与支持

如有问题或建议，请提交 Issue 或 Pull Request。

---

## 实现计划（开发参考）

### Phase 1: 核心功能
- [x] 添加 `analyze` 子命令到 src/cli.py
- [x] 实现模型对比函数
- [x] 实现指标计算
- [x] 实现 JSON 输出

### Phase 2: 可视化
- [x] 添加图表生成
- [x] 实现 HTML 报告
- [x] 添加 CSV 导出

### Phase 3: 高级功能
- [ ] 多模型批量对比
- [ ] 自定义指标
- [ ] 热力图可视化
