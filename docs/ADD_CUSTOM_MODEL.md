# 添加自定义模型指南

本指南说明如何向 `od-benchmark` 添加自定义目标检测模型。

## 方式 1: 使用 Ultralytics 模型（最简单）

如果你的自定义模型是基于 Ultralytics YOLO 的，可以直接使用 `UltralyticsWrapper`：

### 步骤

1. **在 config.yaml 中添加模型配置**：

```yaml
models:
  - name: your_custom_model
    framework: ultralytics
    weights: your_custom_model.pt
    url: https://github.com/user/repo/releases/download/v1.0/your_custom_model.pt
```

2. **直接运行**：

```bash
od-benchmark benchmark --model your_custom_model
```

### 支持的 Ultralytics 模型

任何基于 Ultralytics YOLO 的模型都可以直接使用，包括但不限于：
- YOLOv5 变体
- YOLOv6/YOLOv7/YOLOv8/YOLOv9/YOLOv10
- RT-DETR

只需在 config.yaml 中配置即可！

---

## 方式 2: 创建自定义模型类（高级）

如果你的模型不使用 Ultralytics，需要创建自定义模型类。

### 步骤 1: 创建模型文件

在 `src/models/` 目录下创建新文件，例如 `src/models/my_custom_model.py`：

```python
from typing import List, Dict, Any, Optional
import numpy as np
from .base import BaseModel, Detection


class MyCustomModel(BaseModel):
    def __init__(self, device: str = "auto", conf_threshold: float = 0.001):
        super().__init__(device, conf_threshold)
        self.model_type = "MyCustomModel"

    def load_model(self, weights_path: Optional[str] = None) -> None:
        """加载模型权重"""
        import torch

        if weights_path is None:
            # 使用内置预训练权重
            self.model = torch.hub.load(
                "repo_owner/model_name",
                "model_name",
                pretrained=True
            )
        else:
            # 加载自定义权重
            self.model = torch.load(weights_path, map_location=self.device)

        self.model.to(self.device)
        self.model.eval()

        self.model_info = {
            "name": "MyCustomModel",
            "weights": weights_path,
            "device": self.device,
        }

    def predict(
        self, image: np.ndarray, conf_threshold: Optional[float] = None
    ) -> List[Detection]:
        """执行推理"""
        import torch

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        # 预处理图片
        # 根据你的模型进行调整
        image_tensor = self._preprocess_image(image)

        # 推理
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # 后处理结果
        detections = self._postprocess_outputs(outputs, conf)

        return detections

    def get_model_info(self) -> Dict[str, Any]:
        """返回模型信息"""
        if self.model is None:
            return {}

        info = self.model_info.copy()
        info["params"] = sum(p.numel() for p in self.model.parameters()) / 1e6
        return info

    def warmup(self, image_size: tuple = (640, 640)) -> None:
        """模型预热"""
        import torch
        dummy_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        self.predict(dummy_image)

    def _preprocess_image(self, image: np.ndarray):
        """图片预处理"""
        import torch

        # 根据你的模型实现预处理
        # 示例：调整大小、归一化等
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return transform(image_rgb).unsqueeze(0).to(self.device)

    def _postprocess_outputs(self, outputs, conf_threshold: float):
        """后处理模型输出"""
        import torch

        detections = []

        # 根据你的模型实现后处理
        # 示例：假设输出格式为 [boxes, scores, labels]
        boxes = outputs[0].cpu().numpy()
        scores = outputs[1].cpu().numpy()
        labels = outputs[2].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score >= conf_threshold:
                detections.append(Detection(
                    bbox=box.tolist(),
                    confidence=float(score),
                    class_id=int(label)
                ))

        return detections
```

### 步骤 2: 注册模型

修改 `src/models/__init__.py` 以支持你的新模型：

```python
from typing import Any, Optional
from .base import BaseModel
from .faster_rcnn import FasterRCNN
from .ultralytics_wrapper import UltralyticsWrapper
from .my_custom_model import MyCustomModel  # 导入你的模型

ULTRALYTICS_MODELS = ["yolov8", "yolov9", "yolov10", "rtdetr", "rt-detr"]


def create_model(
    model_name: str, device: str = "auto", conf_threshold: float = 0.001
) -> BaseModel:
    model_key = model_name.lower()

    # 检查 Ultralytics 模型
    for prefix in ULTRALYTICS_MODELS:
        if model_key.startswith(prefix):
            wrapper = UltralyticsWrapper(device, conf_threshold)
            wrapper.model_type = model_name
            return wrapper

    # 检查 FasterRCNN
    if model_key.startswith("faster"):
        return FasterRCNN(device, conf_threshold)

    # 添加你的自定义模型检查
    if model_key.startswith("my_custom"):
        return MyCustomModel(device, conf_threshold)

    raise ValueError(f"Unsupported model: {model_name}")


def load_model_wrapper(
    model: BaseModel, weights_path: Optional[str], model_name: str
) -> None:
    if isinstance(model, UltralyticsWrapper):
        if weights_path is None:
            raise ValueError("Ultralytics models require weights_path")
        model.load_model(weights_path, model_name)
    elif isinstance(model, FasterRCNN):
        model.load_model(weights_path)
    elif isinstance(model, MyCustomModel):  # 添加你的模型加载
        model.load_model(weights_path)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
```

### 步骤 3: 在 config.yaml 中配置

```yaml
models:
  - name: my_custom_model
    framework: custom
    weights: my_custom_model.pt
    url: https://github.com/user/repo/releases/download/v1.0/my_custom_model.pt
```

### 步骤 4: 运行测试

```bash
od-benchmark benchmark --model my_custom_model --num-images 10
```

---

## 方式 3: 扩展 UltralyticsWrapper（中等复杂度）

如果你的自定义模型使用类似 YOLO 的接口，可以扩展 `UltralyticsWrapper`。

### 示例

```python
# src/models/my_yolo_variant.py
from .ultralytics_wrapper import UltralyticsWrapper


class MyYOLOVariant(UltralyticsWrapper):
    """自定义 YOLO 变体，使用 Ultralytics 包装器"""

    def __init__(self, device: str = "auto", conf_threshold: float = 0.001):
        super().__init__(device, conf_threshold)
        self.model_type = "MyYOLOVariant"

    def predict(self, image, conf_threshold=None):
        # 可以在这里添加自定义的后处理逻辑
        detections = super().predict(image, conf_threshold)

        # 自定义处理（可选）
        # detections = self.custom_postprocess(detections)

        return detections
```

然后在 `src/models/__init__.py` 中注册：
```python
from .my_yolo_variant import MyYOLOVariant

def create_model(model_name: str, device: str = "auto", conf_threshold: float = 0.001) -> BaseModel:
    model_key = model_name.lower()

    # ... 其他检查 ...

    if model_key.startswith("my_yolo_variant"):
        return MyYOLOVariant(device, conf_threshold)

    # ... 其他检查 ...
```

---

## 完整示例：添加 YOLOv11 模型

### 1. 创建模型文件

`src/models/yolov11.py`:

```python
from .ultralytics_wrapper import UltralyticsWrapper


class YOLOv11(UltralyticsWrapper):
    """YOLOv11 模型包装器"""

    def __init__(self, device: str = "auto", conf_threshold: float = 0.001):
        super().__init__(device, conf_threshold)
        self.model_type = "YOLOv11"
```

### 2. 注册模型

修改 `src/models/__init__.py`：

```python
from .yolov11 import YOLOv11

ULTRALYTICS_MODELS = ["yolov8", "yolov9", "yolov10", "yolov11", "rtdetr", "rt-detr"]

def create_model(model_name: str, device: str = "auto", conf_threshold: float = 0.001) -> BaseModel:
    model_key = model_name.lower()

    for prefix in ULTRALYTICS_MODELS:
        if model_key.startswith(prefix):
            return YOLOv11(device, conf_threshold)  # 或者使用 wrapper

    # ... 其他检查 ...
```

### 3. 配置模型

`config.yaml`:

```yaml
models:
  - name: yolov11n
    framework: ultralytics
    weights: yolov11n.pt
    url: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov11n.pt
  - name: yolov11s
    framework: ultralytics
    weights: yolov11s.pt
    url: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov11s.pt
```

---

## 关键要点

### 1. 必须实现的接口

自定义模型必须实现 `BaseModel` 的所有抽象方法：

- `load_model(weights_path)` - 加载模型权重
- `predict(image, conf_threshold)` - 执行推理，返回 `List[Detection]`
- `get_model_info()` - 返回模型信息（参数量、大小等）
- `warmup(image_size)` - 模型预热

### 2. 返回格式

`predict()` 方法必须返回 `List[Detection]`，其中：

```python
Detection(
    bbox=[x1, y1, x2, y2],  # 边界框坐标
    confidence=0.95,            # 置信度 (0-1)
    class_id=1                   # 类别 ID (整数)
)
```

### 3. 设备支持

模型应该支持 `device="auto"`，自动选择 CUDA/MPS/CPU：

```python
from .base import BaseModel

class MyModel(BaseModel):
    def __init__(self, device: str = "auto", conf_threshold: float = 0.001):
        super().__init__(device, conf_threshold)
        # 父类会自动处理 device="auto"
```

---

## 常见问题

### Q: 我的模型权重文件应该放在哪里？

A: 放在 `models_cache/` 目录下，或者提供下载 URL 在 config.yaml 中。

### Q: 如何测试我的模型？

A:
```bash
# 下载权重
python scripts/download_weights.py

# 运行测试
od-benchmark benchmark --model my_custom_model --num-images 10
```

### Q: 如何添加类别名称映射？

A: 在 `load_model()` 中设置：

```python
def load_model(self, weights_path):
    # ... 加载模型 ...

    self.names = {
        0: "person",
        1: "car",
        # ...
    }
```

### Q: 我的模型使用不同的输入格式？

A: 在 `_preprocess_image()` 方法中处理：

```python
def _preprocess_image(self, image):
    # 根据你的模型实现预处理
    # 例如：调整大小、归一化、颜色空间转换等
    return processed_image
```

---

## 推荐的工作流程

1. **先实现基本功能** - 确保 `load_model()` 和 `predict()` 可以工作
2. **测试单个模型** - 使用 `--num-images 5` 快速测试
3. **添加可视化支持** - 设置 `self.names` 以便显示类别名称
4. **完整测试** - 使用完整数据集运行
5. **性能优化** - 根据需要优化推理速度

---

## 下一步

参考现有模型实现：
- `src/models/faster_rcnn.py` - 非 Ultralytics 模型示例
- `src/models/ultralytics_wrapper.py` - Ultralytics 包装器示例
- `src/models/base.py` - 基类定义

查看这些文件可以帮助你更好地理解如何实现自定义模型。
