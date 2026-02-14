#!/usr/bin/env python3
"""
快速测试模型对比分析功能 - 多baseline 支持

这个脚本演示如何使用 od-benchmark analyze 命令进行多个基准模型的对比分析。
"""

import sys
from pathlib import Path


def main():
    print("=" * 70)
    print("模型对比分析 - 多baseline 支持测试")
    print("=" * 70)
    print()
    print("这个脚本演示多 baseline 模型支持。")
    print()
    print("使用方式:")
    print("1. 单 baseline vs 单用户模型:")
    print(
        "   od-benchmark analyze --baseline yolov8n --user-model yolov8s --num-images 10"
    )
    print()
    print("2. 多 baseline vs 单用户模型:")
    print(
        "   od-benchmark analyze --baseline yolov8n --baseline yolov8s --user-model yolov8m --num-images 10"
    )
    print()
    print("3. 所有 baseline vs 单用户模型 (--all-baselines):")
    print(
        "   od-benchmark analyze --all-baselines --user-model yolov8n:simulated --num-images 10"
    )
    print()
    print("4. 单 baseline vs 多用户模型:")
    print(
        "   od-benchmark analyze --baseline yolov8n --user-model yolov8s --user-model yolov8m --num-images 10"
    )
    print()
    print("5. 所有 baseline vs 多用户模型:")
    print(
        "   od-benchmark analyze --all-baselines --user-model yolov8s --user-model yolov8m --num-images 50"
    )
    print()
    print("=" * 70)
    print("快速测试：运行 --all-baselines 模式")
    print("=" * 70)
    print()
    print("命令:")
    print(
        "od-benchmark analyze --all-baselines --user-model yolov8n:simulated --num-images 10 --debug"
    )
    print()
    print("说明:")
    print("- --all-baselines: 使用所有配置的基准模型")
    print("- --user-model: 用户自定义模型（这里使用模拟模式）")
    print("- --num-images 10: 只测试 10 张图片（快速）")
    print("- --debug: 启用调试模式")
    print()
    print("输出:")
    print("- 输出格式: JSON + HTML + CSV（全部）")
    print("- 输出目录: outputs/analysis/")
    print("- 汇总文件: summary.json（所有对比汇总）")
    print("- 每个对比: outputs/analysis/comparison_000/, comparison_001/, ...")
    print()
    print("预期结果:")
    print("- 对比 yolov8n yolov8s yolov8m vs yolov8n:simulated")
    print("- 每个 baseline 生成一个独立的对比结果")
    print("- 自动生成汇总报告")
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
        "--all-baselines",
        "--user-model",
        "yolov8n:simulated",
        "--num-images",
        "10",
        "--debug",
        "--format",
        "json",
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
