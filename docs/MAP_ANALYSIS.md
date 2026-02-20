# YOLOv8n mAP 分析报告

## 问题描述

**当前测试结果**：
- mAP@0.50: 0.0000
- mAP@0.50:0.95: 0.0000
- FPS: 39.85

**公开项目标准结果**（YOLOv8n on COCO val2017）：
- mAP@0.50: ~0.373 (37.3%)
- mAP@0.50:0.95: ~0.370 (37.0%)
- FPS: ~80+ (TensorRT) / ~40+ (PyTorch)

**问题**：当前项目的 mAP 远低于公开项目的标准值。

---

## 根本原因分析

### 1. **缺少 NMS（非极大值抑制）处理** ⚠️ **关键问题**

**问题描述**：
- ONNX 模型直接返回所有 8400 个 anchor 的预测结果
- 没有进行 NMS 处理，导致大量重复检测
- 同一个目标被多个 anchor 检测到

**证据**：
```python
# src/models/onnx_model.py:372-436
for i in range(num_anchors):  # 遍历所有 8400 个 anchors
    pred = predictions[i]
    # ... 处理每个 anchor
    if conf < conf_threshold:
        continue
    # 直接添加到 detections 列表，没有 NMS
    detections.append(Detection(...))
```

**影响**：
- 大量重复的检测框
- COCO 指标计算时，检测框过多导致匹配失败
- mAP 极低或为 0

**对比**：
- **PyTorch 模型**：使用 Ultralytics 的 `model.predict()`，**自动进行 NMS**
  ```python
  # src/models/ultralytics_wrapper.py:31
  results = self.model(image, verbose=False, conf=conf)
  # Ultralytics 内部已经执行了 NMS
  # result.boxes 是经过 NMS 过滤的结果
  ```

- **ONNX 模型**：直接返回所有 anchor 预测，**没有 NMS**
  ```python
  # src/models/onnx_model.py:372
  for i in range(num_anchors):
      # 没有任何 NMS 处理
      detections.append(...)
  ```

**修复方案**：
```python
from torchvision.ops import nms

def apply_nms(detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
    """应用 NMS 到检测结果"""
    if not detections:
        return detections
    
    # 转换为 tensor
    boxes = torch.tensor([det.bbox for det in detections])
    scores = torch.tensor([det.confidence for det in detections])
    
    # 应用 NMS
    keep = nms(boxes, scores, iou_threshold)
    
    # 返回过滤后的检测结果
    return [detections[i] for i in keep]
```

---

### 2. **置信度阈值设置不当**

**问题描述**：
- 当前使用 `conf_threshold=0.001` 进行完整 mAP 评估
- 但对于 ONNX 模型，由于没有 NMS，低置信度阈值会导致大量噪声

**证据**：
```python
# benchmark.py:418
conf_threshold = args.conf_threshold or eval_config.get("conf_threshold", 0.001)
```

**影响**：
- 置信度阈值过低时（0.001），会保留大量低质量检测
- 配合缺少 NMS 的问题，会导致大量重复的检测框
- COCO 指标计算时，检测框质量差，导致 mAP 低

**对比**：
- **公开项目**：通常使用 `conf=0.25` 或更高进行可视化
- **完整评估**：使用 `conf=0.001` 进行完整 mAP 计算，但前提是有 NMS

**修复方案**：
1. **立即修复**：在 ONNX 后处理中添加 NMS
2. **参数调整**：在添加 NMS 后，可以使用较低的置信度阈值（0.001）
3. **文档说明**：明确说明 ONNX 模型需要 NMS 处理

---

### 3. **YOLOv8 输出格式解析可能不正确**

**问题描述**：
- YOLOv8 ONNX 输出格式可能不是 `[84, 8400]`
- 可能需要 transpose 或其他处理

**证据**：
```python
# src/models/onnx_model.py:338-341
if shape[1] > 1000:  # 第二维很大，是 YOLOv8/v9 格式 [84, 8400]
    predictions = predictions.T  # 转置为 [8400, 84]
```

**潜在问题**：
1. ONNX 模型输出可能是 `[1, 84, 8400]` 而不是 `[84, 8400]`
2. transpose 操作可能不正确
3. 需要验证实际的输出格式

**验证方法**：
```python
import onnxruntime as ort
session = ort.InferenceSession("models_export/yolov8n.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# 打印输入输出形状
print(f"Input shape: {session.get_inputs()[0].shape}")
print(f"Output shape: {session.get_outputs()[0].shape}")
```

**修复方案**：
1. 验证 ONNX 模型的实际输出格式
2. 根据实际格式调整解析逻辑
3. 添加格式验证和错误提示

---

### 4. **COCO 指标计算可能有问题**

**问题描述**：
- COCO 指标计算可能没有正确处理检测框
- 或者 bbox 格式转换有误

**证据**：
```python
# src/metrics/coco_metrics.py:64-86
def predictions_to_coco_format(self, all_detections: Dict[int, List[Detection]]):
    predictions = []
    for image_id, detections in all_detections.items():
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            width = x2 - x1
            height = y2 - y1
            pred = {
                "image_id": image_id,
                "category_id": det.class_id,
                "bbox": [x1, y1, width, height],  # COCO 格式
                "score": det.confidence,
                "id": ann_id,
            }
```

**潜在问题**：
1. COCO bbox 格式是 `[x, y, width, height]`，与检测框格式一致
2. 但需要确认 `image_id` 和 `category_id` 是否正确
3. 需要验证 `category_id` 是否从 1 开始（COCO 从 1 开始）

**验证方法**：
```python
# 检查检测结果
print(f"总检测框数量: {len(all_detections)}")
print(f"图片数量: {len(all_detections)}")

# 检查每个图片的检测框
for image_id, detections in all_detections.items():
    print(f"图片 {image_id}: {len(detections)} 个检测框")
    for det in detections[:5]:  # 只打印前 5 个
        print(f"  bbox: {det.bbox}, conf: {det.confidence}, class: {det.class_id}")
```

**修复方案**：
1. 验证 bbox 格式是否正确
2. 验证 `category_id` 是否正确（应该是 1-80，不是 0-79）
3. 添加调试输出，打印检测框信息

---

### 5. **类别 ID 映射可能错误**

**问题描述**：
- Ultralytics 模型使用 `cls + 1` 作为 `category_id`
- 但 ONNX 模型可能直接使用原始类别 ID

**证据**：
```python
# src/models/ultralytics_wrapper.py:40
cls = int(box.cls.cpu().numpy()[0])
detections.append(Detection(xyxy, conf, cls + 1))  # +1 转换为 COCO 类别

# src/models/onnx_model.py:381-382
class_id = int(np.argmax(class_scores))
detections.append(Detection(..., class_id=int(class_id)))  # 直接使用，没有 +1
```

**影响**：
- ONNX 模型的类别 ID 可能是 0-79
- COCO 的类别 ID 是 1-80
- 类别 ID 不匹配会导致 mAP 为 0

**修复方案**：
```python
# src/models/onnx_model.py:434
detections.append(
    Detection(
        bbox=[...],
        confidence=float(conf),
        class_id=int(class_id) + 1,  # +1 转换为 COCO 类别
    )
)
```

---

## 综合分析

### 主要问题（按优先级）

| 优先级 | 问题 | 影响 | 修复难度 |
|--------|------|------|----------|
| **P0** | **缺少 NMS 处理** | mAP 为 0 | 中 |
| **P0** | **类别 ID 不匹配** | mAP 为 0 | 低 |
| **P1** | 置信度阈值设置 | mAP 低 | 低 |
| **P2** | ONNX 输出格式解析 | 可能出错 | 中 |
| **P3** | COCO 指标计算 | 需要验证 | 低 |

### 问题根因

**核心问题**：ONNX 模型后处理不完整

1. **没有 NMS**：导致大量重复检测
2. **类别 ID 错误**：类别 ID 未转换为 COCO 格式（1-80）
3. **低置信度阈值**：配合上述问题，导致大量噪声

---

## 修复计划

### 立即修复（P0）

#### 1. 添加 NMS 处理

**文件**：`src/models/onnx_model.py`

**修改内容**：
```python
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from .base import BaseModel, Detection

class ONNXModel(BaseModel):
    def __init__(self, device: str = "auto", conf_threshold: float = 0.001, iou_threshold: float = 0.45):
        super().__init__(device, conf_threshold)
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.names = {}
        self._is_yolo = False
        self.iou_threshold = iou_threshold  # 新增：NMS IOU 阈值

    def _apply_nms(
        self, detections: List[Detection], iou_threshold: float = 0.45
    ) -> List[Detection]:
        """应用 NMS 到检测结果"""
        if not detections:
            return detections
        
        import torch
        
        # 转换为 tensor
        boxes = torch.tensor([det.bbox for det in detections], dtype=torch.float32)
        scores = torch.tensor([det.confidence for det in detections], dtype=torch.float32)
        
        # 应用 NMS
        keep = torch.ops.torchvision.nms(boxes, scores, iou_threshold)
        
        # 返回过滤后的检测结果
        return [detections[i] for i in keep]

    def _postprocess_yolov8(
        self, predictions: np.ndarray, conf_threshold: float, original_shape: tuple
    ) -> List[Detection]:
        """
        处理 YOLOv8/v9 格式的 ONNX 输出
        格式: [num_anchors, 84]
        其中 84 = 4 (bbox: x, y, w, h) + 80 (class scores)
        """
        detections = []
        
        # ... 现有处理逻辑 ...
        
        # 在返回前应用 NMS
        detections = self._apply_nms(detections, self.iou_threshold)
        
        return detections
```

#### 2. 修复类别 ID 映射

**文件**：`src/models/onnx_model.py`

**修改内容**：
```python
# src/models/onnx_model.py:434
detections.append(
    Detection(
        bbox=[
            float(x1_scaled),
            float(y1_scaled),
            float(x2_scaled),
            float(y2_scaled),
        ],
        confidence=float(conf),
        class_id=int(class_id) + 1,  # +1 转换为 COCO 类别（1-80）
    )
)
```

#### 3. 同样修复 _postprocess_yolov10

**文件**：`src/models/onnx_model.py`

**修改内容**：
```python
# src/models/onnx_model.py:503-509
detections.append(
    Detection(
        bbox=[...],
        confidence=float(conf),
        class_id=int(class_id) + 1,  # +1 转换为 COCO 类别
    )
)

# 在返回前应用 NMS
detections = self._apply_nms(detections, self.iou_threshold)
return detections
```

---

## 验证方法

### 1. 验证 NMS 效果

```python
# 统计 NMS 前后的检测框数量
print(f"NMS 前: {len(raw_detections)} 个检测框")
print(f"NMS 后: {len(nms_detections)} 个检测框")
print(f"过滤比例: {(1 - len(nms_detections) / len(raw_detections)) * 100:.1f}%")
```

### 2. 验证类别 ID

```python
# 检查类别 ID 范围
class_ids = [det.class_id for det in detections]
print(f"类别 ID 范围: {min(class_ids)} - {max(class_ids)}")
print(f"应该为 1-80（COCO 格式）")
```

### 3. 验证 mAP

```bash
# 修复后重新测试
mamba run -n benchmark python3 benchmark.py --all --format onnx --num-images 100

# 检查结果
cat outputs/results/yolov8n_result.json | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"mAP@0.50: {data['coco_metrics']['AP@0.50']:.4f}\")
print(f\"mAP@0.50:0.95: {data['coco_metrics']['AP@0.50:0.95']:.4f}\")
"
```

---

## 预期效果

### 修复前
```
mAP@0.50: 0.0000
mAP@0.50:0.95: 0.0000
```

### 修复后
```
mAP@0.50: ~0.373 (接近公开结果）
mAP@0.50:0.95: ~0.370 (接近公开结果）
```

### FPS 影响

- **NMS 开销**：约 10-20% FPS 降低
- **修复前**：39.85 FPS
- **修复后**：约 32-35 FPS
- **可接受**：因为修复前 mAP 为 0，没有实际价值

---

## 总结

**主要问题**：
1. ⚠️ **缺少 NMS 处理**：导致大量重复检测，mAP 为 0
2. ⚠️ **类别 ID 错误**：未转换为 COCO 格式（1-80）
3. ⚠️ **置信度阈值过低**：配合上述问题，导致大量噪声

**修复优先级**：
1. **P0（立即）**：添加 NMS 处理
2. **P0（立即）**：修复类别 ID 映射（+1）
3. **P1（后续）**：验证 ONNX 输出格式
4. **P2（后续）**：优化置信度阈值

**预期效果**：
- mAP@0.50: 从 0.0000 → ~0.373（接近公开结果 37.3%）
- mAP@0.50:0.95: 从 0.0000 → ~0.370（接近公开结果 37.0%）
- FPS: 从 39.85 → ~32-35（可接受的开销）

---

## 下一步

**是否需要立即执行修复计划？**
- 修复 1：添加 NMS 处理
- 修复 2：修复类别 ID 映射
- 修复 3：同样修复 _postprocess_yolov10

**验证步骤**：
1. 运行单元测试
2. 运行端到端测试（10 张图片）
3. 验证 mAP 是否接近公开结果
4. 验证 FPS 是否可接受
