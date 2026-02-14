"""
示例：添加 YOLOv11 模型

这个文件展示了如何通过扩展 UltralyticsWrapper 来添加新的模型支持。
"""

from .ultralytics_wrapper import UltralyticsWrapper


class YOLOv11(UltralyticsWrapper):
    """
    YOLOv11 模型包装器

    由于 YOLOv11 使用与 YOLOv8/YOLOv9/YOLOv10 相同的 Ultralytics 接口，
    我们可以简单地继承 UltralyticsWrapper 并设置 model_type。
    """

    def __init__(self, device: str = "auto", conf_threshold: float = 0.001):
        super().__init__(device, conf_threshold)
        self.model_type = "YOLOv11"
