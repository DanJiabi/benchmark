#!/usr/bin/env python3
"""
基准测试示例 - benchmark_example.py

展示如何使用 od-benchmark 进行模型性能基准测试
"""

import subprocess
import sys


def example_1_single_model():
    """示例 1: 测试单个模型"""
    print("=" * 80)
    print("示例 1: 测试单个模型 (YOLOv8n)")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "benchmark",
        "--model",
        "yolov8n",
        "--num-images",
        "10",
        "--conf-threshold",
        "0.25",
        "--output-dir",
        "outputs/examples/single_model",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 测试 YOLOv8n 模型")
    print("  - 使用 10 张图片进行快速测试")
    print("  - 置信度阈值设置为 0.25")
    print("  - 结果保存到 outputs/examples/single_model")
    print()


def example_2_multiple_models():
    """示例 2: 测试多个模型"""
    print("=" * 80)
    print("示例 2: 测试多个模型对比")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "benchmark",
        "--model",
        "yolov8n",
        "--model",
        "yolov8s",
        "--model",
        "yolov10n",
        "--num-images",
        "50",
        "--output-dir",
        "outputs/examples/multi_model",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 同时测试 YOLOv8n、YOLOv8s、YOLOv10n")
    print("  - 使用 50 张图片进行对比")
    print("  - 自动生成对比图表")
    print()


def example_3_all_models():
    """示例 3: 测试所有模型"""
    print("=" * 80)
    print("示例 3: 测试所有配置的模型")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "benchmark",
        "--all",
        "--num-images",
        "100",
        "--conf-threshold",
        "0.001",
        "--output-dir",
        "outputs/examples/all_models",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 测试 config.yaml 中配置的所有模型")
    print("  - 使用 100 张图片进行全面评估")
    print("  - 低置信度阈值 (0.001) 用于完整的 mAP 评估")
    print()


def example_4_with_visualization():
    """示例 4: 带可视化的基准测试"""
    print("=" * 80)
    print("示例 4: 带检测框可视化的测试")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "benchmark",
        "--model",
        "yolov8n",
        "--num-images",
        "20",
        "--visualize",
        "--num-viz-images",
        "10",
        "--conf-threshold",
        "0.25",
        "--output-dir",
        "outputs/examples/with_viz",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 启用可视化 (--visualize)")
    print("  - 可视化前 10 张图片的检测结果")
    print("  - 检测框图片保存到 outputs/examples/with_viz/visualizations/")
    print()


def example_5_custom_config():
    """示例 5: 使用自定义配置"""
    print("=" * 80)
    print("示例 5: 使用自定义配置文件")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "benchmark",
        "--config",
        "config.yaml",
        "--model",
        "yolov8n",
        "--num-images",
        "10",
        "--output-dir",
        "outputs/examples/custom_config",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 使用指定的配置文件")
    print("  - 可以修改 config.yaml 来自定义模型列表和参数")
    print()


def example_6_different_conf_thresholds():
    """示例 6: 不同置信度阈值对比"""
    print("=" * 80)
    print("示例 6: 不同置信度阈值的影响")
    print("=" * 80)

    print("\n低阈值 (0.001) - 用于完整评估:")
    cmd1 = [
        "od-benchmark",
        "benchmark",
        "--model",
        "yolov8n",
        "--num-images",
        "10",
        "--conf-threshold",
        "0.001",
        "--output-dir",
        "outputs/examples/low_conf",
    ]
    print(f"  {' '.join(cmd1)}")

    print("\n中阈值 (0.25) - 用于可视化:")
    cmd2 = [
        "od-benchmark",
        "benchmark",
        "--model",
        "yolov8n",
        "--num-images",
        "10",
        "--conf-threshold",
        "0.25",
        "--output-dir",
        "outputs/examples/med_conf",
    ]
    print(f"  {' '.join(cmd2)}")

    print("\n高阈值 (0.5) - 只检测高置信度目标:")
    cmd3 = [
        "od-benchmark",
        "benchmark",
        "--model",
        "yolov8n",
        "--num-images",
        "10",
        "--conf-threshold",
        "0.5",
        "--output-dir",
        "outputs/examples/high_conf",
    ]
    print(f"  {' '.join(cmd3)}")
    print()


def run_examples():
    """运行所有示例（仅打印命令，不实际执行）"""
    print("\n" + "=" * 80)
    print("OD-Benchmark 基准测试示例")
    print("=" * 80)
    print("\n以下示例展示了如何使用 od-benchmark 进行模型性能测试。")
    print("这些命令可以直接在终端中运行。\n")

    example_1_single_model()
    example_2_multiple_models()
    example_3_all_models()
    example_4_with_visualization()
    example_5_custom_config()
    example_6_different_conf_thresholds()

    print("=" * 80)
    print("提示")
    print("=" * 80)
    print("""
1. 首次运行前，确保已下载模型权重:
   python scripts/download_weights.py

2. 如果使用 Apple Silicon (M1/M2/M3/M4)，设置环境变量:
   export PYTORCH_ENABLE_MPS_FALLBACK=1

3. 查看所有可用选项:
   od-benchmark benchmark --help

4. 测试结果保存在 outputs/examples/ 目录下，包括:
   - 结果表格 (CSV/JSON)
   - 性能对比图表
   - 可视化图片（如果启用 --visualize）
""")


if __name__ == "__main__":
    run_examples()
