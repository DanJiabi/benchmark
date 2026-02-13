#!/usr/bin/env python3
"""
下载 config.yaml 中配置的模型权重文件
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, Any
import yaml
import requests


def get_file_hash(file_path: Path, algorithm: str = "md5") -> str:
    """计算文件的哈希值"""
    hash_func = getattr(hashlib, algorithm)()
    chunk_size = 8192
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def get_expected_hash(url: str) -> str:
    """从 GitHub 获取文件哈希值"""
    try:
        # GitHub API 获取文件信息
        if "github.com" in url:
            api_url = url.replace(
                "https://github.com/", "https://api.github.com/repos/"
            ).replace("/releases/download/", "/releases/assets/")

            # 尝试获取文件列表
            headers = {"Accept": "application/vnd.github.v3+json"}
            response = requests.get(api_url, headers=headers, timeout=10)
            if response.status_code == 200:
                # 返回 None，因为我们无法直接获取哈希
                return None

        return None
    except Exception:
        return None


def get_min_expected_size(file_name: str) -> int:
    """根据文件名获取最小预期大小（字节）"""
    file_name = file_name.lower()

    # YOLO 模型文件大小参考
    if "yolov8n" in file_name:
        return 5 * 1024 * 1024  # 5 MB
    elif "yolov8s" in file_name:
        return 10 * 1024 * 1024  # 10 MB
    elif "yolov8m" in file_name:
        return 20 * 1024 * 1024  # 20 MB
    elif "yolov8l" in file_name:
        return 40 * 1024 * 1024  # 40 MB
    elif "yolov8x" in file_name:
        return 60 * 1024 * 1024  # 60 MB
    elif "yolov9t" in file_name:
        return 5 * 1024 * 1024  # 5 MB
    elif "yolov9s" in file_name:
        return 10 * 1024 * 1024  # 10 MB
    elif "yolov9m" in file_name:
        return 20 * 1024 * 1024  # 20 MB
    elif "yolov10n" in file_name:
        return 5 * 1024 * 1024  # 5 MB
    elif "yolov10s" in file_name:
        return 10 * 1024 * 1024  # 10 MB
    elif "yolov10m" in file_name:
        return 20 * 1024 * 1024  # 20 MB
    elif "yolov10b" in file_name:
        return 30 * 1024 * 1024  # 30 MB
    elif "rtdetr-l" in file_name:
        return 50 * 1024 * 1024  # 50 MB
    elif "rtdetr-x" in file_name:
        return 100 * 1024 * 1024  # 100 MB

    # 默认最小大小：至少 100KB
    return 100 * 1024


def check_file_complete(file_path: Path, expected_size: int = None) -> tuple[bool, str]:
    """检查文件是否完整"""
    if not file_path.exists():
        return False, "文件不存在"

    file_size = file_path.stat().st_size

    # 检查文件是否为空
    if file_size == 0:
        return False, "文件为空"

    # 检查文件是否过小（不完整）
    min_size = get_min_expected_size(file_path.name)
    if file_size < min_size:
        size_mb = file_size / 1024 / 1024
        min_mb = min_size / 1024 / 1024
        return False, f"文件过小 ({size_mb:.2f} MB < {min_mb:.2f} MB), 可能下载不完整"

    # 检查文件大小是否匹配预期
    if expected_size and file_size < expected_size * 0.95:
        return False, f"文件大小不匹配: {file_size} < {expected_size}"

    # 检查文件是否可以读取
    try:
        with open(file_path, "rb") as f:
            data = f.read(1024)
            # 检查是否是有效的模型文件（PyTorch checkpoint 或其他格式）
            if len(data) < 10:
                return False, "文件内容异常"
    except Exception as e:
        return False, f"文件读取失败: {e}"

    # 尝试加载验证（可选，需要安装 torch）
    try:
        import torch

        try:
            # 尝试加载文件（仅加载权重）
            checkpoint = torch.load(file_path, map_location="cpu", weights_only=True)
            # 检查加载的内容是否合理
            if isinstance(checkpoint, dict):
                # 检查是否包含典型的 PyTorch checkpoint 键
                valid_keys = ["model", "state_dict", "ema", "model_state_dict"]
                has_valid_key = any(key in checkpoint for key in valid_keys)
                if has_valid_key or len(checkpoint) > 0:
                    return True, "文件完整且可加载"
            # 对于 ULPALYCS 格式的模型
            elif hasattr(checkpoint, "model"):
                return True, "文件完整且可加载"
        except Exception as load_error:
            error_str = str(load_error)
            # 过滤掉 PyTorch 2.6 的警告信息
            if "Weights only load failed" in error_str:
                try:
                    # 尝试使用 weights_only=False
                    checkpoint = torch.load(
                        file_path, map_location="cpu", weights_only=False
                    )
                    return True, "文件完整且可加载"
                except Exception:
                    return False, "文件加载失败"
            else:
                # 其他加载错误
                return False, f"文件加载失败: {load_error}"
    except ImportError:
        # 未安装 torch，跳过加载验证
        pass

    return True, "文件完整"


def download_file(
    url: str,
    output_path: Path,
    expected_size: int = None,
    overwrite: bool = False,
) -> bool:
    """下载文件"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 检查文件是否已存在且完整
    if not overwrite and output_path.exists():
        is_complete, message = check_file_complete(output_path, expected_size)
        if is_complete:
            print(f"  ✅ 文件已存在且完整: {output_path.name}")
            return True
        else:
            print(f"  ⚠️  文件不完整 ({message}), 将重新下载")

    print(f"  📥 下载: {url}")
    print(f"  📁 保存到: {output_path}")

    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded_size = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)

                    # 显示进度
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(
                            f"\r  ⏳ 进度: {progress:.1f}% ({downloaded_size}/{total_size})",
                            end="",
                        )

        print()  # 换行

        # 验证下载的文件
        if total_size > 0 and downloaded_size != total_size:
            print(f"  ⚠️  下载大小不匹配: {downloaded_size}/{total_size}")
            output_path.unlink()
            return False

        print(
            f"  ✅ 下载完成: {output_path.name} ({downloaded_size / 1024 / 1024:.2f} MB)"
        )
        return True

    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载配置文件"""
    config_file = Path(__file__).parent / config_path

    if not config_file.exists():
        print(f"❌ 配置文件不存在: {config_file}")
        sys.exit(1)

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def download_models(
    config_path: str,
    models_cache: str = "models_cache",
    overwrite: bool = False,
) -> None:
    """下载配置文件中的所有模型权重"""
    print("=" * 80)
    print("模型权重下载工具")
    print("=" * 80)

    config = load_config(config_path)

    if "models" not in config:
        print("❌ 配置文件中没有找到 'models' 节")
        sys.exit(1)

    models = config["models"]
    cache_dir = Path(models_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n配置文件: {config_path}")
    print(f"缓存目录: {cache_dir}")
    print(f"模型数量: {len(models)}")
    print("=" * 80)

    success_count = 0
    fail_count = 0
    skip_count = 0
    incomplete_count = 0

    results = []

    for idx, model_config in enumerate(models, 1):
        print(f"\n[{idx}/{len(models)}] 处理模型")
        print("-" * 80)

        model_name = model_config.get("name", "unknown")
        weights = model_config.get("weights")
        url = model_config.get("url")

        if weights is None:
            print(f"  ℹ️  {model_name}: 无权重文件（可能使用内置预训练权重）")
            skip_count += 1
            results.append((model_name, "skip", "无权重文件"))
            continue

        if not url:
            print(f"  ⚠️  {model_name}: 未提供下载 URL")
            fail_count += 1
            results.append((model_name, "fail", "未提供 URL"))
            continue

        weights_path = cache_dir / weights

        # 检查文件是否需要下载
        need_download = overwrite

        if not overwrite and weights_path.exists():
            is_complete, message = check_file_complete(weights_path)
            if not is_complete:
                print(f"  ⚠️  {message}")
                need_download = True
                incomplete_count += 1
        else:
            if not overwrite:
                print(f"  🔍 文件不存在: {weights}")
                need_download = True
                incomplete_count += 1

        # 下载文件
        if need_download:
            success = download_file(url, weights_path)
            if success:
                success_count += 1
                results.append((model_name, "download", "下载成功"))
            else:
                fail_count += 1
                results.append((model_name, "fail", "下载失败"))
        else:
            skip_count += 1
            results.append((model_name, "skip", "文件已存在且完整"))

    # 打印汇总
    print("\n" + "=" * 80)
    print("下载汇总")
    print("=" * 80)
    print(f"  总模型数: {len(models)}")
    print(f"  下载成功: {success_count}")
    print(f"  重新下载（不完整）: {incomplete_count}")
    print(f"  跳过（已存在）: {skip_count}")
    print(f"  下载失败: {fail_count}")
    print("=" * 80)

    # 保存结果到文件
    results_file = cache_dir / "download_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("模型权重下载结果\n")
        f.write("=" * 80 + "\n\n")
        for model_name, status, message in results:
            f.write(f"{model_name}: {status} - {message}\n")

    print(f"结果已保存: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="下载 config.yaml 中的模型权重文件")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径（默认: config.yaml）",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="models_cache",
        help="缓存目录路径（默认: models_cache）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的文件",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="仅检查文件完整性，不下载",
    )

    args = parser.parse_args()

    if args.check_only:
        # 仅检查模式
        print("=" * 80)
        print("检查模型权重文件完整性")
        print("=" * 80)

        config = load_config(args.config)
        cache_dir = Path(args.cache_dir)

        if "models" not in config:
            print("❌ 配置文件中没有找到 'models' 节")
            sys.exit(1)

        models = config["models"]

        for idx, model_config in enumerate(models, 1):
            model_name = model_config.get("name", "unknown")
            weights = model_config.get("weights")

            if weights is None:
                continue

            weights_path = cache_dir / weights

            if weights_path.exists():
                is_complete, message = check_file_complete(weights_path)
                status = "✅ 完整" if is_complete else "❌ 不完整"
                print(
                    f"[{idx}/{len(models)}] {model_name:20s} {status:10s} - {message}"
                )
            else:
                print(f"[{idx}/{len(models)}] {model_name:20s} ⚠️  不存在")
    else:
        # 下载模式
        download_models(args.config, args.cache_dir, args.overwrite)


if __name__ == "__main__":
    main()
