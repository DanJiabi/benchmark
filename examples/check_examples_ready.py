#!/usr/bin/env python3
"""
示例验证脚本 - 检查环境是否就绪

运行: python examples/check_examples_ready.py
"""

import sys
import subprocess
from pathlib import Path


def check_command(cmd, name):
    """检查命令是否可用"""
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ {name}: 可用")
            return True
        else:
            print(f"❌ {name}: 返回错误")
            return False
    except Exception as e:
        print(f"❌ {name}: 不可用 - {e}")
        return False


def check_file(path, name):
    """检查文件是否存在"""
    if Path(path).exists():
        print(f"✅ {name}: 存在")
        return True
    else:
        print(f"❌ {name}: 不存在")
        return False


def main():
    print("=" * 60)
    print("OD-Benchmark 示例环境检查")
    print("=" * 60)
    print()

    all_ok = True

    # 1. 检查核心命令
    print("1. 检查核心命令:")
    all_ok &= check_command("odb --help", "odb 命令")
    print()

    # 2. 检查示例文件
    print("2. 检查示例文件:")
    examples = [
        ("examples/quick_start.py", "快速开始"),
        ("examples/benchmark_example.py", "基准测试示例"),
        ("examples/analyze_example.py", "对比分析示例"),
        ("examples/export_example.py", "导出示例"),
        ("examples/compare_example.py", "格式对比示例"),
        ("examples/COMPARE_PT_ONNX.md", "ONNX对比文档"),
    ]
    for path, name in examples:
        all_ok &= check_file(path, name)
    print()

    # 3. 检查数据集
    print("3. 检查数据集:")
    dataset_path = Path.home() / "raw/COCO/val2017"
    if dataset_path.exists():
        jpg_count = len(list(dataset_path.glob("*.jpg")))
        print(f"✅ COCO数据集: 存在 ({jpg_count} 张图片)")
    else:
        print(f"❌ COCO数据集: 不存在")
        print(f"   期望路径: {dataset_path}")
        all_ok = False
    print()

    # 4. 检查模型权重
    print("4. 检查模型权重:")
    cache_path = Path("models_cache")
    if cache_path.exists():
        pt_count = len(list(cache_path.glob("*.pt")))
        print(f"✅ 模型缓存: 存在 ({pt_count} 个 .pt 文件)")
    else:
        print(f"⚠️  模型缓存: 不存在")
        print(f"   运行: python scripts/download_weights.py")
    print()

    # 5. 总结
    print("=" * 60)
    if all_ok:
        print("✅ 环境检查通过！可以运行示例。")
        print()
        print("推荐开始:")
        print("  python examples/quick_start.py")
    else:
        print("⚠️  部分检查未通过，请参考上方提示修复。")
        print()
        print("常见问题:")
        print("  1. 如果 odb 命令不可用，运行: pip install -e .")
        print("  2. 如果数据集不存在，修改 config.yaml 中的路径")
        print("  3. 如果没有模型权重，运行: python scripts/download_weights.py")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
