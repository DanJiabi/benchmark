"""
Object Detection Performance Benchmark

A comprehensive benchmarking framework for evaluating object detection models
on the COCO dataset.
"""

__version__ = "0.1.0"

from .models import create_model, load_model_wrapper
from .models.base import BaseModel, Detection

__all__ = [
    "__version__",
    "create_model",
    "load_model_wrapper",
    "BaseModel",
    "Detection",
]
