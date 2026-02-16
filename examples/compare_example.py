#!/usr/bin/env python3
"""
格式性能对比示例 - compare_example.py

展示如何使用 od-benchmark compare 对比不同格式的性能
"""

import subprocess
import sys


def example_1_basic_comparison():
    """示例 1: 基础格式对比"""
    print("=" * 80)
    print("示例 1: PyTorch vs ONNX 基础对比")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "compare",
        "--model",
        "models_cache/yolov8n.pt",
        "--num-images",
        "50",
        "--output-dir",
        "outputs/examples/compare/basic",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 对比 PyTorch 和 ONNX 两种格式")
    print("  - 自动导出 ONNX 模型（如果不存在）")
    print("  - 测试推理速度和精度")
    print("  - 生成性能对比报告")
    print()


def example_2_pytorch_only():
    """示例 2: 只测试 PyTorch"""
    print("=" * 80)
    print("示例 2: 只测试 PyTorch 格式")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "compare",
        "--model",
        "models_cache/yolov8n.pt",
        "--formats",
        "pytorch",
        "--num-images",
        "50",
        "--output-dir",
        "outputs/examples/compare/pytorch_only",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 只测试 PyTorch 格式")
    print("  - 用于基准性能测试")
    print("  - 不导出 ONNX 模型")
    print()


def example_3_different_num_images():
    """示例 3: 不同测试图片数量"""
    print("=" * 80)
    print("示例 3: 不同测试图片数量的影响")
    print("=" * 80)

    print("\n快速测试 (10 张图片):")
    cmd1 = [
        "od-benchmark",
        "compare",
        "--model",
        "models_cache/yolov8n.pt",
        "--num-images",
        "10",
        "--output-dir",
        "outputs/examples/compare/quick",
    ]
    print(f"  {' '.join(cmd1)}")

    print("\n中等精度 (50 张图片):")
    cmd2 = [
        "od-benchmark",
        "compare",
        "--model",
        "models_cache/yolov8n.pt",
        "--num-images",
        "50",
        "--output-dir",
        "outputs/examples/compare/medium",
    ]
    print(f"  {' '.join(cmd2)}")

    print("\n完整测试 (500 张图片):")
    cmd3 = [
        "od-benchmark",
        "compare",
        "--model",
        "models_cache/yolov8n.pt",
        "--num-images",
        "500",
        "--output-dir",
        "outputs/examples/compare/full",
    ]
    print(f"  {' '.join(cmd3)}")
    print()


def example_4_custom_model_name():
    """示例 4: 自定义模型名称"""
    print("=" * 80)
    print("示例 4: 使用自定义模型名称")
    print("=" * 80)

    cmd = [
        "od-benchmark",
        "compare",
        "--model",
        "models_cache/yolov8n.pt",
        "--model-name",
        "My Custom YOLO",
        "--num-images",
        "50",
        "--output-dir",
        "outputs/examples/compare/custom_name",
    ]

    print(f"命令: {' '.join(cmd)}")
    print("\n说明:")
    print("  - 为模型设置自定义显示名称")
    print("  - 适用于测试自己训练的模型")
    print("  - 报告中使用自定义名称")
    print()


def example_5_batch_comparison():
    """示例 5: 批量对比多个模型"""
    print("=" * 80)
    print("示例 5: 批量对比多个模型")
    print("=" * 80)

    script = """#!/bin/bash

# 批量对比多个模型的 PyTorch vs ONNX 性能
MODELS=("yolov8n" "yolov8s" "yolov8m" "yolov10n")
NUM_IMAGES=50

for model in "${MODELS[@]}"; do
    echo "======================================"
    echo "对比模型: $model"
    echo "======================================"
    
    od-benchmark compare \\
        --model "models_cache/${model}.pt" \\
        --num-images $NUM_IMAGES \\
        --output-dir "outputs/examples/compare/batch/${model}"
    
    echo ""
done

echo "所有对比完成！"
echo "报告位置: outputs/examples/compare/batch/"
"""

    print("批量对比脚本:")
    print(script)
    print()


def example_6_comparison_with_export():
    """示例 6: 先导出再对比"""
    print("=" * 80)
    print("示例 6: 先导出 ONNX 模型再对比")
    print("=" * 80)

    print("\n步骤 1: 导出 ONNX 模型")
    cmd_export = [
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
        "models_export",
    ]
    print(f"  {' '.join(cmd_export)}")

    print("\n步骤 2: 对比性能")
    cmd_compare = [
        "od-benchmark",
        "compare",
        "--model",
        "models_cache/yolov8n.pt",
        "--num-images",
        "50",
        "--output-dir",
        "outputs/examples/compare/exported",
    ]
    print(f"  {' '.join(cmd_compare)}")

    print("\n说明:")
    print("  - 先手动导出 ONNX 模型（可以使用自定义参数）")
    print("  - 然后使用 od-benchmark compare 对比性能")
    print("  - 导出的模型会被自动使用，不会重复导出")
    print()


def example_7_performance_summary():
    """示例 7: 生成性能摘要"""
    print("=" * 80)
    print("示例 7: 提取性能摘要")
    print("=" * 80)

    script = """#!/usr/bin/env python3
\"\"\"
从多个对比报告中提取性能摘要
\"\"\"

import json
import glob
from pathlib import Path

# 查找所有对比报告
report_files = glob.glob("outputs/examples/compare/batch/*/*_format_comparison.json")

print("=" * 80)
print("格式性能对比摘要")
print("=" * 80)
print()

for report_file in report_files:
    with open(report_file) as f:
        data = json.load(f)
    
    model_name = data['model_name']
    pytorch_fps = data['formats']['pytorch']['fps']
    onnx_fps = data['formats']['onnx']['fps']
    speedup = data['speed_comparison']['speedup']
    speedup_pct = data['speed_comparison']['speedup_pct']
    
    print(f"模型: {model_name}")
    print(f"  PyTorch FPS: {pytorch_fps:.2f}")
    print(f"  ONNX FPS:    {onnx_fps:.2f}")
    print(f"  加速比:      {speedup:.2f}x ({speedup_pct:+.1f}%)")
    print()

print("=" * 80)
"""

    print("性能摘要脚本:")
    print(script)
    print()


def run_examples():
    """运行所有示例"""
    print("\n" + "=" * 80)
    print("OD-Benchmark 格式性能对比示例")
    print("=" * 80)
    print("\n以下示例展示了如何使用 od-benchmark compare 对比不同格式的性能。\n")

    example_1_basic_comparison()
    example_2_pytorch_only()
    example_3_different_num_images()
    example_4_custom_model_name()
    example_5_batch_comparison()
    example_6_comparison_with_export()
    example_7_performance_summary()

    print("=" * 80)
    print("输出文件说明")
    print("=" * 80)
    print("""
对比完成后，在输出目录中会生成以下文件:

1. {model_name}.onnx
   - 导出的 ONNX 模型文件
   - 包含完整的模型结构
   - 可用于部署

2. {model_name}_format_comparison.json
   - 详细的对比报告
   - 包含 FPS、mAP 等性能指标
   - 适合程序化处理

3. 控制台输出
   - 实时对比进度
   - 性能差异摘要
   - 推荐意见
""")

    print("=" * 80)
    print("对比指标说明")
    print("=" * 80)
    print("""
性能指标:
  - FPS: 每秒帧数，越高越好
  - 平均推理时间: 单张图片的推理时间（毫秒），越低越好
  - 加速比: ONNX FPS / PyTorch FPS

精度指标:
  - mAP@0.50: IoU 阈值 0.50 时的平均精度
  - mAP@0.50:0.95: COCO 主指标
  - 精度差异: PyTorch mAP - ONNX mAP

推荐逻辑:
  - 速度提升 > 10%: ✅ 建议使用 ONNX
  - 速度提升 ±10%: ⚖️ 性能相近，按需选择
  - 速度提升 < -10%: ⚠️ 建议使用 PyTorch
  - 精度损失 < 1%: ✅ 可忽略
  - 精度损失 1-3%: ⚖️ 可接受范围
  - 精度损失 > 3%: ⚠️ 需要检查
""")

    print("=" * 80)
    print("Apple Silicon 特别说明")
    print("=" * 80)
    print("""
在 Apple Silicon (M1/M2/M3/M4) 上:

1. PyTorch MPS:
   - 通常是最快的后端
   - 约比 ONNX CoreML 快 2-3 倍
   - 推荐优先使用

2. ONNX Runtime:
   - 可以使用 CoreML ExecutionProvider
   - 比纯 CPU 快约 50%
   - 适合需要跨平台部署的场景

3. 环境变量:
   export PYTORCH_ENABLE_MPS_FALLBACK=1

注意: Apple Silicon 上 ONNX 比 PyTorch MPS 慢是正常的，
这不代表 ONNX 性能差，而是 PyTorch MPS 优化得很好。
""")

    print("=" * 80)
    print("注意事项")
    print("=" * 80)
    print("""
1. 首次运行会自动导出 ONNX 模型（5-30秒）

2. ONNX 模型会被缓存，下次直接使用

3. ONNX 推理使用简化的后处理
   - 主要关注 FPS 性能指标
   - mAP 可能为 0 或不准确（正常现象）

4. 批处理大小
   - 当前使用 batch_size=1
   - 如需测试更大 batch，需修改代码

5. 导出失败
   - 检查 ultralytics 安装: pip install --upgrade ultralytics
   - 检查 onnxruntime 安装: pip install onnxruntime
""")


if __name__ == "__main__":
    run_examples()
