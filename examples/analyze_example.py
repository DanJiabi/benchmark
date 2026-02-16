#!/usr/bin/env python3
"""
模型对比分析示例 - analyze_example.py

展示如何使用 od-benchmark analyze 进行模型对比分析
"""

import subprocess
import sys


def example_1_basic_comparison():
    """示例 1: 基础模型对比"""
    print("=" * 80)
    print("示例 1: 基础模型对比 (YOLOv8n vs YOLOv8s)")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "analyze",
        "--baseline",
        "yolov8n",
        "--user-model",
        "yolov8s",
        "--num-images",
        "50",
        "--format",
        "all",
        "--output-dir",
        "outputs/examples/analysis/basic",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 以 YOLOv8n 作为基准模型")
    print("  - 对比 YOLOv8s 的性能差异")
    print("  - 生成 JSON、HTML、CSV 三种格式的报告")
    print("  - 包含 mAP、FPS、模型大小等全面对比")
    print()


def example_2_all_baselines():
    """示例 2: 对比所有基准模型"""
    print("=" * 80)
    print("示例 2: 对比所有基准模型")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "analyze",
        "--all-baselines",
        "--user-model",
        "yolov10n",
        "--num-images",
        "100",
        "--format",
        "html",
        "--output-dir",
        "outputs/examples/analysis/all_baselines",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 使用 config.yaml 中配置的所有模型作为基准")
    print("  - 将 YOLOv10n 与所有基准模型对比")
    print("  - 生成交互式 HTML 报告")
    print()


def example_3_custom_model_weights():
    """示例 3: 对比自定义权重文件"""
    print("=" * 80)
    print("示例 3: 对比自定义权重文件")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "analyze",
        "--baseline",
        "yolov8n",
        "--user-model",
        "models_cache/my_custom_model.pt",
        "--num-images",
        "50",
        "--format",
        "all",
        "--output-dir",
        "outputs/examples/analysis/custom",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 基准模型: YOLOv8n")
    print("  - 用户模型: 自定义权重文件路径")
    print("  - 适用于对比自己训练的模型")
    print()


def example_4_different_formats():
    """示例 4: 不同输出格式对比"""
    print("=" * 80)
    print("示例 4: 不同输出格式的对比")
    print("=" * 80)

    print("\nJSON 格式 (适合程序处理):")
    cmd1 = [
        "od-benchmark",
        "analyze",
        "--baseline",
        "yolov8n",
        "--user-model",
        "yolov8s",
        "--format",
        "json",
        "--output-dir",
        "outputs/examples/analysis/json_format",
    ]
    print(f"  {' '.join(cmd1)}")

    print("\nHTML 格式 (适合人工查看):")
    cmd2 = [
        "od-benchmark",
        "analyze",
        "--baseline",
        "yolov8n",
        "--user-model",
        "yolov8s",
        "--format",
        "html",
        "--output-dir",
        "outputs/examples/analysis/html_format",
    ]
    print(f"  {' '.join(cmd2)}")

    print("\nCSV 格式 (适合数据分析):")
    cmd3 = [
        "od-benchmark",
        "analyze",
        "--baseline",
        "yolov8n",
        "--user-model",
        "yolov8s",
        "--format",
        "csv",
        "--output-dir",
        "outputs/examples/analysis/csv_format",
    ]
    print(f"  {' '.join(cmd3)}")

    print("\nAll 格式 (生成所有格式):")
    cmd4 = [
        "od-benchmark",
        "analyze",
        "--baseline",
        "yolov8n",
        "--user-model",
        "yolov8s",
        "--format",
        "all",
        "--output-dir",
        "outputs/examples/analysis/all_formats",
    ]
    print(f"  {' '.join(cmd4)}")
    print()


def example_5_debug_mode():
    """示例 5: 调试模式"""
    print("=" * 80)
    print("示例 5: 调试模式")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "analyze",
        "--baseline",
        "yolov8n",
        "--user-model",
        "yolov8s",
        "--num-images",
        "10",
        "--debug",
        "--output-dir",
        "outputs/examples/analysis/debug",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 启用调试模式 (--debug)")
    print("  - 输出详细的调试信息")
    print("  - 适用于排查问题")
    print()


def run_examples():
    """运行所有示例（仅打印命令）"""
    print("\n" + "=" * 80)
    print("OD-Benchmark 模型对比分析示例")
    print("=" * 80)
    print("\n以下示例展示了如何使用 od-benchmark analyze 进行模型对比。\n")

    example_1_basic_comparison()
    example_2_all_baselines()
    example_3_custom_model_weights()
    example_4_different_formats()
    example_5_debug_mode()

    print("=" * 80)
    print("输出文件说明")
    print("=" * 80)
    print("""
分析完成后，在输出目录中会生成以下文件:

1. comparison.json
   - 包含所有对比指标的详细数据
   - 适合程序自动化处理

2. comparison.html
   - 交互式 HTML 报告
   - 包含表格、图表、推荐意见
   - 可以直接在浏览器中打开

3. comparison.csv
   - 表格格式的对比结果
   - 适合导入 Excel 进行进一步分析

4. 控制台输出
   - 实时的对比进度
   - 关键的性能差异
   - 推荐意见
""")

    print("=" * 80)
    print("对比指标说明")
    print("=" * 80)
    print("""
准确性指标:
  - mAP@0.50: IoU 阈值 0.50 时的平均精度
  - mAP@0.50:0.95: COCO 主指标，IoU 0.50-0.95 的平均 mAP
  - AP@small/medium/large: 不同尺寸目标的 AP

性能指标:
  - FPS: 每秒帧数 (Frames Per Second)
  - 平均推理时间: 单张图片的推理时间 (毫秒)
  - 加速比: 用户模型速度 / 基准模型速度

模型信息:
  - 参数量: 模型参数数量 (百万)
  - 模型大小: 权重文件大小 (MB)
""")


if __name__ == "__main__":
    run_examples()
