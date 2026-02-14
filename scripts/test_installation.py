#!/usr/bin/env python3
"""
测试脚本 - 验证 pyproject.toml 和 od-benchmark 命令是否正常工作
"""

import sys
from pathlib import Path


def test_basic_imports():
    """测试基本的包导入"""
    print("测试 1: 基本包导入")
    try:
        from src.models import create_model, load_model_wrapper
        print("  ✅ src.models 导入成功")
    except Exception as e:
        print(f"  ❌ src.models 导入失败: {e}")
        return False

    try:
        from src.analysis import ModelComparison
        print("  ✅ src.analysis 导入成功")
    except Exception as e:
        print(f"  ❌ src.analysis 导入失败: {e}")
        return False

    return True


def test_cli_command():
    """测试 CLI 命令"""
    print("\n测试 2: CLI 命令")
    import subprocess

    # 测试 --help
    result = subprocess.run(
        ["od-benchmark", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if "od-benchmark" in result.stdout:
        print("  ✅ od-benchmark --help 正常工作")
    else:
        print(f"  ❌ od-benchmark --help 失败")
        print(f"stdout: {result.stdout[:200]}")
        return False

    # 测试 benchmark --help
    result = subprocess.run(
        ["od-benchmark", "benchmark", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if "usage" in result.stdout:
        print("  ✅ od-benchmark benchmark --help 正常工作")
    else:
        print(f"  ❌ od-benchmark benchmark --help 失败")
        print(f"stdout: {result.stdout[:200]}")
        return False

    # 测试 analyze --help
    result = subprocess.run(
        ["od-benchmark", "analyze", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if "Compare baseline" in result.stdout:
        print("  ✅ od-benchmark analyze --help 正常工作")
    else:
        print(f"  ❌ od-benchmark analyze --help 失败")
        print(f"stdout: {result.stdout[:200]}")
        return False

    return True


def test_analyze_command():
    """测试 analyze 命令"""
    print("\n测试 3: analyze 命令")
    import subprocess

    # 测试 --help
    result = subprocess.run(
        ["od-benchmark", "analyze", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if "Compare baseline model" in result.stdout:
        print("  ✅ od-benchmark analyze --help 正常工作")
    else:
        print(f"  ❌ od-benchmark analyze --help 失败")
        return False

    # 测试 --all-baselines 参数
    result = subprocess.run(
        ["od-benchmark", "analyze", "--all-baselines", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if "--all-baselines" in result.stdout:
        print("  ✅ od-benchmark analyze --all-baselines --help 正常工作")
        else:
            print(f"  ❌ od-benchmark analyze --all-baselines --help 失败")
            return False

    # 测试参数解析
    result = subprocess.run(
        ["od-benchmark", "analyze", "--baseline", "yolov8n", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if "--baseline" in result.stdout:
        print("  ✅ 参数解析正常")
    else:
        print(f"  ❌ 参数解析失败")
            return False

    # 测试 --user-model 参数
    result = subprocess.run(
        ["od-benchmark", "analyze", "--user-model", "test_model", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    if "--user-model" in result.stdout:
        print("  ✅ --user-model 参数解析正常")
        else:
        print(f"  ❌ --user-model 参数解析失败")
            return False

    return True


def test_main():
    """主测试函数"""
    print("=" * 70)
    print("od-benchmark 安装验证测试")
    print("=" * 70)
    print()

    results = []

    # 测试 1: 基本导入
    results.append(("基本导入", test_basic_imports()))

    # 测试 2: CLI 命令
    results.append(("CLI命令", test_cli_command()))

    # 测试 3: analyze 命令
    results.append(("analyze命令", test_analyze_command()))

    # 汇总
    print("\n")
    print("=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    print()

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:25s} {status}")

    print()
    print("=" * 70)

    all_passed = all(result for _, passed in results)

    if all_passed:
        print("✅ 所有测试通过！")
        return 0
    else:
        print("❌ 有测试失败")
        print("请检查错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(test_main())
