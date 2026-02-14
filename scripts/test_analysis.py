#!/usr/bin/env python3
"""
快速测试模型对比分析功能

这个脚本演示如何使用 od-benchmark analyze 命令进行模型对比分析。
"""

import sys
from pathlib import Path


def main():
    print("=" * 70)
    print("模型对比分析 - 快速演示")
    print("=" * 70)
    print()
    print("这个脚本演示如何使用模型对比分析功能。")
    print()
    print("使用方式:")
    print("1. 模拟模式（快速演示）:")
    print(
        "   od-benchmark analyze --baseline yolov8n --user-model yolov8n:simulated --num-images 10 --debug"
    )
    print()
    print("2. 对比两个标准模型:")
    print(
        "   od-benchmark analyze --baseline yolov8n --user-model yolov8s --num-images 100"
    )
    print()
    print("3. 对比基准模型和自定义权重文件:")
    print(
        "   od-benchmark analyze --baseline yolov8n --user-model path/to/custom.pt --num-images 50"
    )
    print()
    print("4. 使用用户模型配置文件:")
    print(
        "   od-benchmark analyze --baseline yolov8n --user-model user_models/my_model.yaml --num-images 100 --format all"
    )
    print()
    print("=" * 70)
    print("快速测试：运行模拟模式")
    print("=" * 70)
    print()
    print("命令:")
    print(
        "od-benchmark analyze --baseline yolov8n --user-model yolov8n:simulated --num-images 10 --debug"
    )
    print()
    print("说明:")
    print("- 基准模型: yolov8n（真实的 YOLOv8n 模型）")
    print("- 用户模型: yolov8n:simulated（模拟的自定义模型）")
    print("- 测试图片: 10 张")
    print("- 调试模式: 启用（显示详细信息）")
    print()
    print("结果:")
    print("- 输出格式: JSON + HTML + CSV（全部）")
    print("- 输出目录: outputs/analysis/")
    print()
    print("模拟模型特性:")
    print("- 降低 15% 置信度（模拟不够准确）")
    print("- 其他指标与基准模型相同")
    print()


def run_quick_test():
    """运行快速测试"""
    import subprocess
    import time

    print("运行快速测试...")
    print()

    cmd = [
        "od-benchmark",
        "analyze",
        "--baseline",
        "yolov8n",
        "--user-model",
        "yolov8n:simulated",
        "--num-images",
        "10",
        "--debug",
    ]

    print(f"执行命令: {' '.join(cmd)}")
    print()

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10分钟超时
        )

        elapsed = time.time() - start_time

        print("=" * 70)
        print("快速测试完成")
        print("=" * 70)
        print()

        if result.returncode == 0:
            print("✅ 测试成功完成")
            print(f"⏱️  用时: {elapsed:.2f} 秒")
            print()
            print("输出:")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        else:
            print("❌ 测试失败")
            print(f"返回码: {result.returncode}")
            print()
            print("错误输出:")
            print(result.stderr)

    except subprocess.TimeoutExpired:
        print("❌ 测试超时（超过 10 分钟）")
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

    print()
    print("是否运行快速测试？")
    print("  按 Enter 键运行快速测试")
    print("  或按 Ctrl+C 取消")
    print()

    try:
        input()
        run_quick_test()
    except KeyboardInterrupt:
        print("\n已取消测试")
