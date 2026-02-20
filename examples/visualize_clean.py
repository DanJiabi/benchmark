#!/usr/bin/env python3
"""
目标检测模型推理结果可视化 - 支持多模型对比
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import yaml

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base import Detection
from src.models import create_model, load_model_wrapper
from src.utils.visualization import draw_detection_boxes
from src.utils import resolve_model_path
from examples.utils import load_model_for_demo

import cv2
import numpy as np


def get_models_from_config(config_path: str = "config.yaml") -> List[str]:
    """从 config.yaml 读取所有模型名称"""
    config_file = Path(__file__).parent.parent / config_path

    if not config_file.exists():
        return []

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if "models" not in config:
            return []

        models = []
        for model_config in config["models"]:
            if "name" in model_config:
                models.append(model_config["name"])

        return models

    except Exception:
        return []


DEFAULT_MODELS = ["yolov8n", "faster_rcnn"]

COCO_VAL_PATH = "~/raw/COCO/val2017"
OUTPUT_BASE_DIR = "outputs/visualizations"
CONFIDENCE_THRESHOLD = 0.25
CONFIG_FILE = "config.yaml"

TEST_IMAGES = [
    "000000139077.jpg",  # 包含人物、电视、遥控器
    "000000139260.jpg",  # 包含蛋糕、香蕉
    "000000139871.jpg",  # 包含飞机
    "000000139872.jpg",  # 包含狗、飞盘
    "000000000285.jpg",  # 包含多人、自行车
    "000000000632.jpg",  # 包含汽车
    "000000001268.jpg",  # 包含人物、手提包、鸟、船
    "000000001296.jpg",  # 其他场景
    "000000001353.jpg",  # 其他场景
    "000000001425.jpg",  # 其他场景
]


def get_class_names(model) -> Dict[int, str]:
    """获取模型的类别名称映射"""
    try:
        if hasattr(model, "model") and hasattr(model.model, "names"):
            return model.model.names
        elif hasattr(model, "names"):
            return model.names
        else:
            return {}
    except Exception:
        return {}


def draw_detections(
    image: np.ndarray,
    detections: List[Detection],
    class_names: Dict[int, str],
    model_name: str,
    max_boxes: int = 10,
) -> tuple:
    """绘制检测框到图片上"""
    img_draw = draw_detection_boxes(image, detections, class_names, max_boxes)
    count = min(len(detections), max_boxes)
    return img_draw, count


def process_single_image(
    model, model_name: str, image_path: Path, output_path: Path, conf_threshold: float
) -> Dict[str, Any]:
    """处理单张图片的推理和可视化"""
    result = {
        "success": False,
        "detections_count": 0,
        "drawn_count": 0,
        "error": None,
    }

    try:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            result["error"] = "无法读取图片"
            return result

        h, w = image_bgr.shape[:2]

        detections = model.predict(image_bgr, conf_threshold)
        result["detections_count"] = len(detections)

        class_names = get_class_names(model)

        img_draw, drawn_count = draw_detections(
            image_bgr, detections, class_names, model_name
        )
        result["drawn_count"] = drawn_count

        os.makedirs(output_path.parent, exist_ok=True)
        success = cv2.imwrite(str(output_path), img_draw)

        if success:
            result["success"] = True
        else:
            result["error"] = "保存图片失败"

    except Exception as e:
        result["error"] = str(e)

    return result


def create_comparison_thumbnails(
    comparison_dir: Path,
    model_results: Dict[str, Dict[str, Path]],
    models_to_test: List[str],
    num_images: int,
) -> None:
    """生成模型对比缩略图"""
    if len(models_to_test) < 2:
        return

    print(f"\n{'=' * 80}")
    print("[对比] 生成缩略对比图")
    print("=" * 80)

    comparison_dir.mkdir(parents=True, exist_ok=True)

    for img_idx in range(num_images):
        image_name = TEST_IMAGES[img_idx]

        # 收集所有模型该图片的可视化结果
        model_images = []
        model_labels = []

        for model_name in models_to_test:
            if model_name in model_results and img_idx in model_results[model_name]:
                img_path = model_results[model_name][img_idx]
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        model_images.append(img)
                        model_labels.append(model_name)

        # 如果只有一个或零个模型有结果，跳过
        if len(model_images) < 2:
            continue

        # 生成网格布局
        num_models = len(model_images)
        if num_models == 2:
            cols = 2
            rows = 1
        elif num_models == 3:
            cols = 3
            rows = 1
        else:
            cols = 2
            rows = (num_models + 1) // 2

        # 计算每个图片的缩略图大小
        thumbnail_width = 320
        thumbnail_height = 320

        thumbnails = []
        for img in model_images:
            # 调整大小为缩略图
            thumbnail = cv2.resize(img, (thumbnail_width, thumbnail_height))
            # 添加模型名称标签
            label_img = np.zeros((40, thumbnail_width, 3), dtype=np.uint8)
            label_img[:] = [0, 0, 0]
            model_name = model_labels[len(thumbnails)]
            cv2.putText(
                label_img,
                model_name,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            thumbnail = np.vstack([label_img, thumbnail])
            thumbnails.append(thumbnail)

        # 创建网格
        grid_img = []
        for r in range(rows):
            start_idx = r * cols
            end_idx = min((r + 1) * cols, len(thumbnails))
            row_images = thumbnails[start_idx:end_idx]

            if len(row_images) == 0:
                continue

            if len(row_images) == cols:
                row = np.hstack(row_images)
            else:
                row = np.hstack(row_images)
                # 补齐空位
                if len(row_images) < cols:
                    # 使用第一个图片的形状来创建 padding
                    ref_shape = row_images[0].shape
                    padding = np.zeros(ref_shape, dtype=np.uint8)
                    padding[:] = [50, 50, 50]
                    row = np.hstack([row] + [padding] * (cols - len(row_images)))

            grid_img.append(row)

        if len(grid_img) > 0:
            comparison = np.vstack(grid_img)

            # 保存对比图
            output_path = comparison_dir / f"comparison_{img_idx:02d}_{image_name}"
            cv2.imwrite(str(output_path), comparison)

            print(f"  ✅ 生成对比图: {output_path.name}")

    print(f"  ✅ 对比图保存目录: {comparison_dir}")


def load_model(model_name: str, conf_threshold: float = CONFIDENCE_THRESHOLD) -> Any:
    """加载指定的模型"""
    try:
        model, name = load_model_for_demo(
            model_name, device="auto", conf_threshold=conf_threshold, verbose=False
        )
        print(f"  ✅ 加载模型: {name}")
        return model, name
    except FileNotFoundError as e:
        print(f"  ⚠️  {e}")
        return None, None
    except Exception as e:
        print(f"  ❌ 加载模型失败: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="目标检测模型推理结果可视化 - 支持多模型对比"
    )
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        nargs="+",
        help="指定要使用的模型（可多个，如: --model yolov8n yolov8s）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="使用所有配置的模型",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"置信度阈值（默认: {CONFIDENCE_THRESHOLD}）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_BASE_DIR,
        help=f"输出目录（默认: {OUTPUT_BASE_DIR}）",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="最多处理图片数量",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_FILE,
        help=f"配置文件路径（默认: {CONFIG_FILE}）",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("目标检测模型推理结果可视化")
    print("=" * 80)

    coco_path = Path(COCO_VAL_PATH).expanduser()

    if not coco_path.exists():
        print(f"❌ 数据集路径不存在: {coco_path}")
        return

    # 从配置文件读取默认模型列表
    config_models = get_models_from_config(args.config)

    available_models = list(Path("models_cache").glob("*.pt"))
    available_model_names = [p.stem for p in available_models]
    available_model_names.extend(["faster_rcnn"])

    if args.all:
        models_to_test = available_model_names
    elif args.model:
        models_to_test = [m for sublist in args.model for m in sublist]
    else:
        # 使用配置文件中的模型列表
        if config_models:
            models_to_test = config_models
            print(f"从 {args.config} 读取到 {len(config_models)} 个模型")
        else:
            # 如果配置文件读取失败，使用 yolov8n 和 faster_rcnn
            models_to_test = ["yolov8n", "faster_rcnn"]
            print(f"无法从 {args.config} 读取模型配置，使用默认模型")

    models_to_test = list(set(models_to_test))
    models_to_test.sort()

    print(f"\n选择的模型: {', '.join(models_to_test)}")
    print(f"置信度阈值: {args.conf_threshold}")
    print(f"输出目录: {args.output_dir}")

    num_images = (
        min(len(TEST_IMAGES), args.max_images) if args.max_images else len(TEST_IMAGES)
    )

    print(f"\n将处理 {num_images} 张图片")
    print("=" * 80)

    all_results = {}
    model_output_paths = {}

    for model_name in models_to_test:
        print(f"\n{'=' * 80}")
        print(f"[模型] {model_name}")
        print("=" * 80)

        try:
            model, model_type = load_model(model_name, args.conf_threshold)

            if model is None:
                print(f"  ⚠️  模型 {model_name} 加载失败，跳过")
                continue

            print(f"  模型类型: {model_type}")

            model_output_dir = Path(args.output_dir) / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)

            model_stats = {
                "total_images": 0,
                "total_detections": 0,
                "total_drawn": 0,
                "success_count": 0,
                "fail_count": 0,
            }

            model_output_paths[model_name] = {}

            for idx in range(num_images):
                image_name = TEST_IMAGES[idx]
                image_path = coco_path / image_name
                output_path = model_output_dir / f"detection_{idx:02d}_{image_name}"

                print(f"\n[{idx + 1}/{num_images}] {image_name}")

                result = process_single_image(
                    model, model_name, image_path, output_path, args.conf_threshold
                )

                model_stats["total_images"] += 1
                model_stats["total_detections"] += result["detections_count"]
                model_stats["total_drawn"] += result["drawn_count"]

                if result["success"]:
                    model_stats["success_count"] += 1
                    model_output_paths[model_name][idx] = output_path
                    print(
                        f"  ✅ 检测: {result['detections_count']}, 绘制: {result['drawn_count']}"
                    )
                else:
                    model_stats["fail_count"] += 1
                    print(f"  ❌ 失败: {result['error']}")

            all_results[model_name] = model_stats

            print(f"\n{'=' * 80}")
            print(f"[{model_name}] 汇总")
            print("=" * 80)
            print(
                f"  成功: {model_stats['success_count']}/{model_stats['total_images']}"
            )
            print(f"  失败: {model_stats['fail_count']}/{model_stats['total_images']}")
            print(f"  总检测数: {model_stats['total_detections']}")
            print(f"  总绘制数: {model_stats['total_drawn']}")
            print(
                f"  平均检测/图: {model_stats['total_detections'] / model_stats['total_images']:.2f}"
            )

        except Exception as e:
            print(f"❌ 模型 {model_name} 加载失败: {e}")
            import traceback

            traceback.print_exc()

    # 生成对比缩略图
    if len(models_to_test) > 1:
        comparison_dir = Path(args.output_dir) / "comparison"
        create_comparison_thumbnails(
            comparison_dir, model_output_paths, models_to_test, num_images
        )

    if len(all_results) > 1:
        print(f"\n{'=' * 80}")
        print("[对比] 模型性能对比")
        print("=" * 80)
        print(f"{'模型':<15} {'成功':<10} {'总检测':<10} {'平均/图':<10}")
        print("-" * 50)
        for model_name, stats in all_results.items():
            avg = (
                stats["total_detections"] / stats["total_images"]
                if stats["total_images"] > 0
                else 0
            )
            print(
                f"{model_name:<15} {stats['success_count']:<10} {stats['total_detections']:<10} {avg:<10.2f}"
            )

    print("\n" + "=" * 80)
    print("处理完成！")
    print(f"输出目录: {args.output_dir}")
    if len(models_to_test) > 1:
        print(f"对比图目录: {Path(args.output_dir) / 'comparison'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
