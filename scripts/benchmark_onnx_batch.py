#!/usr/bin/env python3
"""
批量测试所有 ONNX 模型
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_onnx_models(export_dir: str = "models_export") -> list[Path]:
    """查找所有 ONNX 模型"""
    export_path = Path(export_dir)
    if not export_path.exists():
        print(f"❌ 导出目录不存在: {export_dir}")
        print("💡 提示: 先运行 'od-benchmark export --all-models --format onnx'")
        return []

    onnx_files = sorted(export_path.glob("*.onnx"))
    return onnx_files


def benchmark_onnx_models(
    onnx_files: list[Path],
    num_images: int = 50,
    conf_threshold: float = 0.001,
    output_dir: str = "results/onnx_benchmark",
    verbose: bool = True,
) -> dict:
    """批量测试 ONNX 模型"""

    results = {}
    total = len(onnx_files)

    print("=" * 80)
    print(f"批量测试 ONNX 模型")
    print("=" * 80)
    print(f"模型数量: {total}")
    print(f"测试图片数: {num_images}")
    print(f"置信度阈值: {conf_threshold}")
    print(f"输出目录: {output_dir}")
    print("=" * 80)

    for idx, onnx_file in enumerate(onnx_files, 1):
        print(f"\n[{idx}/{total}] 测试: {onnx_file.name}")
        print("-" * 80)

        cmd = [
            "od-benchmark",
            "benchmark",
            "--model",
            str(onnx_file),
            "--num-images",
            str(num_images),
            "--conf-threshold",
            str(conf_threshold),
            "--output-dir",
            output_dir,
        ]

        if verbose:
            print(f"命令: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=not verbose,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                print(f"✅ {onnx_file.name} 测试完成")
                results[str(onnx_file)] = {
                    "success": True,
                    "model": onnx_file.name,
                }
            else:
                print(f"❌ {onnx_file.name} 测试失败")
                if result.stderr:
                    print(f"错误: {result.stderr}")
                results[str(onnx_file)] = {
                    "success": False,
                    "model": onnx_file.name,
                    "error": result.stderr,
                }
        except Exception as e:
            print(f"❌ {onnx_file.name} 测试异常: {e}")
            results[str(onnx_file)] = {
                "success": False,
                "model": onnx_file.name,
                "error": str(e),
            }

    # 汇总
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)

    success_count = sum(1 for r in results.values() if r["success"])
    failed_count = total - success_count

    print(f"✅ 成功: {success_count}/{total}")
    print(f"❌ 失败: {failed_count}/{total}")

    if failed_count > 0:
        print("\n失败的模型:")
        for model_path, result in results.items():
            if not result["success"]:
                print(f"  - {result['model']}: {result.get('error', 'Unknown error')}")

    print(f"\n详细结果保存在: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="批量测试所有 ONNX 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试所有 ONNX 模型（默认 50 张图片）
  python scripts/benchmark_onnx_batch.py
  
  # 测试所有 ONNX 模型（100 张图片）
  python scripts/benchmark_onnx_batch.py --num-images 100
  
  # 指定导出目录
  python scripts/benchmark_onnx_batch.py --export-dir models_export
  
  # 快速测试（10 张图片）
  python scripts/benchmark_onnx_batch.py --num-images 10
  
  # 详细输出
  python scripts/benchmark_onnx_batch.py --verbose
        """,
    )

    parser.add_argument(
        "--export-dir",
        type=str,
        default="models_export",
        help="ONNX 模型目录（默认: models_export）",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="测试图片数量（默认: 50）",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.001,
        help="置信度阈值（默认: 0.001）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/onnx_benchmark",
        help="输出目录（默认: results/onnx_benchmark）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细输出",
    )

    args = parser.parse_args()

    # 查找 ONNX 模型
    onnx_files = find_onnx_models(args.export_dir)

    if not onnx_files:
        print("\n💡 提示: 先运行以下命令导出 ONNX 模型:")
        print("   od-benchmark export --all-models --format onnx")
        sys.exit(1)

    # 批量测试
    results = benchmark_onnx_models(
        onnx_files=onnx_files,
        num_images=args.num_images,
        conf_threshold=args.conf_threshold,
        output_dir=args.output_dir,
        verbose=args.verbose,
    )

    # 返回码
    success_count = sum(1 for r in results.values() if r["success"])
    sys.exit(0 if success_count == len(onnx_files) else 1)


if __name__ == "__main__":
    main()
