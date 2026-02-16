#!/usr/bin/env python3
"""
模型导出示例 - export_example.py

展示如何使用 od-benchmark export 导出模型到不同格式
"""

import subprocess
import sys


def example_1_export_to_onnx():
    """示例 1: 导出为 ONNX 格式"""
    print("=" * 80)
    print("示例 1: 导出为 ONNX 格式")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "export",
        "--model",
        "models_cache/yolov8n.pt",
        "--format",
        "onnx",
        "--input-size",
        "640",
        "640",
        "--simplify",
        "--output-dir",
        "outputs/examples/export",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 将 PyTorch 模型 (.pt) 导出为 ONNX 格式")
    print("  - 输入尺寸: 640x640")
    print("  - 启用模型简化 (--simplify)")
    print("  - 输出: outputs/examples/export/yolov8n.onnx")
    print()


def example_2_export_dynamic_onnx():
    """示例 2: 导出动态尺寸 ONNX"""
    print("=" * 80)
    print("示例 2: 导出动态尺寸 ONNX")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "export",
        "--model",
        "models_cache/yolov8n.pt",
        "--format",
        "onnx",
        "--dynamic",
        "--input-size",
        "640",
        "640",
        "--output-dir",
        "outputs/examples/export",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 启用动态输入尺寸 (--dynamic)")
    print("  - 可以接受不同尺寸的输入图片")
    print("  - 适合需要处理多种分辨率的场景")
    print()


def example_3_export_all_models():
    """示例 3: 批量导出所有模型"""
    print("=" * 80)
    print("示例 3: 批量导出所有模型")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "export",
        "--all-models",
        "--format",
        "onnx",
        "--input-size",
        "640",
        "640",
        "--output-dir",
        "outputs/examples/export/all_models",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 导出 config.yaml 中配置的所有模型")
    print("  - 自动处理每个模型的导出")
    print("  - 批量导出到指定目录")
    print()


def example_4_different_input_sizes():
    """示例 4: 不同输入尺寸导出"""
    print("=" * 80)
    print("示例 4: 不同输入尺寸导出")
    print("=" * 80)

    print("\n移动端优化 (320x320):")
    cmd1 = [
        "od-benchmark",
        "export",
        "--model",
        "models_cache/yolov8n.pt",
        "--format",
        "onnx",
        "--input-size",
        "320",
        "320",
        "--output-dir",
        "outputs/examples/export/mobile",
    ]
    print(f"  {' '.join(cmd1)}")

    print("\n标准尺寸 (640x640):")
    cmd2 = [
        "od-benchmark",
        "export",
        "--model",
        "models_cache/yolov8n.pt",
        "--format",
        "onnx",
        "--input-size",
        "640",
        "640",
        "--output-dir",
        "outputs/examples/export/standard",
    ]
    print(f"  {' '.join(cmd2)}")

    print("\n高精度 (1280x1280):")
    cmd3 = [
        "od-benchmark",
        "export",
        "--model",
        "models_cache/yolov8n.pt",
        "--format",
        "onnx",
        "--input-size",
        "1280",
        "1280",
        "--output-dir",
        "outputs/examples/export/hd",
    ]
    print(f"  {' '.join(cmd3)}")
    print()


def example_5_export_script():
    """示例 5: 批量导出脚本"""
    print("=" * 80)
    print("示例 5: 批量导出脚本示例")
    print("=" * 80)

    script = """#!/bin/bash

# 批量导出多个模型到 ONNX
MODELS=("yolov8n" "yolov8s" "yolov8m")

for model in "${MODELS[@]}"; do
    echo "导出 $model..."
    od-benchmark export \\
        --model "models_cache/${model}.pt" \\
        --format onnx \\
        --input-size 640 640 \\
        --output-dir "models_export/${model}"
done

echo "所有模型导出完成！"
"""

    print("批量导出脚本:")
    print(script)
    print()


def example_6_compare_formats():
    """示例 6: 格式对比"""
    print("=" * 80)
    print("示例 6: 对比 PyTorch 和 ONNX 性能")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "compare",
        "--model",
        "models_cache/yolov8n.pt",
        "--formats",
        "pytorch,onnx",
        "--num-images",
        "50",
        "--output-dir",
        "outputs/examples/format_comparison",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 自动导出 ONNX 模型（如果不存在）")
    print("  - 对比 PyTorch 和 ONNX 的推理速度")
    print("  - 生成性能对比报告")
    print()


def run_examples():
    """运行所有示例"""
    print("\n" + "=" * 80)
    print("OD-Benchmark 模型导出示例")
    print("=" * 80)
    print("\n以下示例展示了如何使用 od-benchmark export 导出模型。\n")

    example_1_export_to_onnx()
    example_2_export_dynamic_onnx()
    example_3_export_all_models()
    example_4_different_input_sizes()
    example_5_export_script()
    example_6_compare_formats()

    print("=" * 80)
    print("导出格式说明")
    print("=" * 80)
    print("""
ONNX 格式:
  - 跨平台通用格式
  - 支持多种推理框架 (ONNX Runtime, OpenVINO, TensorRT)
  - 文件扩展名: .onnx
  - 适合部署到边缘设备

导出选项:
  --simplify: 简化模型结构，减少节点数量
  --dynamic: 支持动态输入尺寸
  --fp16: 使用 FP16 精度（减小模型大小，可能损失精度）
  --int8: 使用 INT8 量化（显著减小模型大小，需要校准）
""")

    print("=" * 80)
    print("使用导出模型的示例代码")
    print("=" * 80)
    print("""
Python 中使用 ONNX 模型:

import onnxruntime as ort
import numpy as np
import cv2

# 加载 ONNX 模型
session = ort.InferenceSession("yolov8n.onnx")
input_name = session.get_inputs()[0].name

# 读取并预处理图片
image = cv2.imread("image.jpg")
image = cv2.resize(image, (640, 640))
image = image[:, :, ::-1]  # BGR to RGB
image = image.transpose(2, 0, 1)  # HWC to CHW
image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0

# 推理
outputs = session.run(None, {input_name: image})

# 后处理（根据模型输出格式）
# ...
""")


if __name__ == "__main__":
    run_examples()
