#!/usr/bin/env python3
"""
使用 ONNX 格式进行基准测试的示例

演示如何使用 --format onnx 参数批量测试所有 ONNX 模型
"""

import subprocess
import sys


def example_1_export_all_models():
    """示例 1: 导出所有模型为 ONNX 格式"""
    print("=" * 80)
    print("示例 1: 导出所有模型为 ONNX 格式")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "export",
        "--all-models",
        "--format",
        "onnx",
        "--output-dir",
        "models_export",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 导出 config.yaml 中配置的所有模型为 ONNX 格式")
    print("  - 使用默认参数（输入尺寸 640x640，启用简化）")
    print("  - ONNX 模型将保存到 models_export/ 目录")
    print()


def example_2_benchmark_all_onnx():
    """示例 2: 测试所有 ONNX 模型"""
    print("=" * 80)
    print("示例 2: 批量测试所有 ONNX 模型")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "benchmark",
        "--all",
        "--format",
        "onnx",
        "--num-images",
        "50",
        "--output-dir",
        "outputs/results/onnx_benchmark",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 测试 models_export/ 目录中的所有 ONNX 模型")
    print("  - 使用 50 张图片进行测试")
    print("  - 置信度阈值为 0.001（用于完整的 mAP 评估）")
    print("  - 结果保存到 outputs/results/onnx_benchmark/")
    print()


def example_3_benchmark_specific_onnx():
    """示例 3: 测试指定的 ONNX 模型"""
    print("=" * 80)
    print("示例 3: 测试指定的 ONNX 模型")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "benchmark",
        "--model",
        "yolov8n.onnx",
        "--model",
        "yolov8s.onnx",
        "--format",
        "onnx",
        "--num-images",
        "100",
        "--output-dir",
        "outputs/results/specific_onnx",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 测试指定的 ONNX 模型（yolov8n 和 yolov8s）")
    print("  - 使用 100 张图片进行更准确的评估")
    print("  - 结果保存到 outputs/results/specific_onnx/")
    print()


def example_4_benchmark_with_visualization():
    """示例 4: 带可视化的 ONNX 模型测试"""
    print("=" * 80)
    print("示例 4: 带可视化的 ONNX 模型测试")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "benchmark",
        "--model",
        "yolov8n.onnx",
        "--format",
        "onnx",
        "--visualize",
        "--num-viz-images",
        "10",
        "--num-images",
        "20",
        "--conf-threshold",
        "0.25",
        "--output-dir",
        "outputs/results/onnx_with_viz",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 测试 yolov8n.onnx 并启用可视化")
    print("  - 可视化前 10 张图片的检测结果")
    print("  - 使用较高的置信度阈值 (0.25) 使结果更清晰")
    print("  - 可视化图片保存到 outputs/visualizations/")
    print()


def example_5_compare_pytorch_vs_onnx():
    """示例 5: 对比 PyTorch 和 ONNX 性能"""
    print("=" * 80)
    print("示例 5: 对比 PyTorch 和 ONNX 性能")
    print("=" * 80)

    print("\n步骤 1: 测试 PyTorch 模型")
    cmd_pytorch = [
        "od-benchmark",
        "benchmark",
        "--all",
        "--format",
        "pytorch",
        "--num-images",
        "50",
        "--output-dir",
        "outputs/results/pytorch_benchmark",
    ]
    print(f"  {' '.join(cmd_pytorch)}")

    print("\n步骤 2: 测试 ONNX 模型")
    cmd_onnx = [
        "od-benchmark",
        "benchmark",
        "--all",
        "--format",
        "onnx",
        "--num-images",
        "50",
        "--output-dir",
        "outputs/results/onnx_benchmark",
    ]
    print(f"  {' '.join(cmd_onnx)}")

    print("\n步骤 3: 对比结果")
    print("  比较两个目录中的结果文件，分析性能差异")
    print()


def example_6_batch_script():
    """示例 6: 批量导出和测试脚本"""
    print("=" * 80)
    print("示例 6: 一键导出并测试所有模型")
    print("=" * 80)

    script = """#!/bin/bash

# 一键导出并测试所有 ONNX 模型

echo "======================================"
echo "步骤 1: 导出所有模型为 ONNX 格式"
echo "======================================"
od-benchmark export --all-models --format onnx

echo ""
echo "======================================"
echo "步骤 2: 测试所有 ONNX 模型"
echo "======================================"
od-benchmark benchmark --all --format onnx --num-images 100

echo ""
echo "======================================"
echo "完成！"
echo "======================================"
echo "结果保存在 outputs/results/ 目录"
"""

    print("脚本内容:")
    print(script)
    print()


def example_7_different_confidence():
    """示例 7: 不同置信度阈值测试"""
    print("=" * 80)
    print("示例 7: 不同置信度阈值的 ONNX 测试")
    print("=" * 80)

    print("\n低阈值（完整 mAP 评估）:")
    cmd1 = [
        "od-benchmark",
        "benchmark",
        "--model",
        "yolov8n.onnx",
        "--format",
        "onnx",
        "--num-images",
        "100",
        "--conf-threshold",
        "0.001",
        "--output-dir",
        "outputs/results/onnx_low_conf",
    ]
    print(f"  {' '.join(cmd1)}")

    print("\n中阈值（可视化友好）:")
    cmd2 = [
        "od-benchmark",
        "benchmark",
        "--model",
        "yolov8n.onnx",
        "--format",
        "onnx",
        "--visualize",
        "--num-images",
        "20",
        "--conf-threshold",
        "0.25",
        "--output-dir",
        "outputs/results/onnx_med_conf",
    ]
    print(f"  {' '.join(cmd2)}")

    print("\n高阈值（仅高置信度检测）:")
    cmd3 = [
        "od-benchmark",
        "benchmark",
        "--model",
        "yolov8n.onnx",
        "--format",
        "onnx",
        "--num-images",
        "100",
        "--conf-threshold",
        "0.5",
        "--output-dir",
        "outputs/results/onnx_high_conf",
    ]
    print(f"  {' '.join(cmd3)}")
    print()


def run_examples():
    """运行所有示例（仅打印命令）"""
    print("\n" + "=" * 80)
    print("ONNX 格式基准测试示例")
    print("=" * 80)
    print("\n以下示例展示了如何使用 --format onnx 参数进行 ONNX 模型测试。\n")

    example_1_export_all_models()
    example_2_benchmark_all_onnx()
    example_3_benchmark_specific_onnx()
    example_4_benchmark_with_visualization()
    example_5_compare_pytorch_vs_onnx()
    example_6_batch_script()
    example_7_different_confidence()

    print("=" * 80)
    print("注意事项")
    print("=" * 80)
    print("""
1. 导出 ONNX 模型:
   - 首次使用前，必须先导出 ONNX 模型
   - 运行: od-benchmark export --all-models --format onnx
   - ONNX 模型保存在 models_export/ 目录

2. 格式说明:
   - --format pytorch: 测试 PyTorch 模型（默认）
   - --format onnx: 测试 ONNX 模型

3. 模型路径:
   - PyTorch: 从 models_cache/ 加载 .pt 文件
   - ONNX: 从 models_export/ 加载 .onnx 文件

4. 性能对比:
   - ONNX 通常比 PyTorch 快 10-30%
   - 精度损失通常 < 1%
   - Apple Silicon 上 PyTorch MPS 仍然最快

5. 快速测试:
   - 使用 --num-images 10 进行快速验证
   - 使用 --num-images 100 进行中等精度测试
   - 使用全部图片（不指定 --num-images）进行完整测试
""")

    print("=" * 80)
    print("常见错误")
    print("=" * 80)
    print("""
错误: ONNX 模型目录不存在: models_export
解决: 先运行 'od-benchmark export --all-models --format onnx'

错误: 在 models_export 中未找到 ONNX 模型
解决: 检查 models_export/ 目录，或重新导出模型

错误: 未找到要测试的 ONNX 模型
解决: 使用 --all 或指定正确的 ONNX 模型名称
""")

    print("=" * 80)
    print("下一步")
    print("=" * 80)
    print("""
1. 导出所有模型为 ONNX 格式
2. 运行批量测试
3. 查看结果和对比图表
4. 对比 PyTorch 和 ONNX 的性能差异
""")


if __name__ == "__main__":
    run_examples()
