#!/usr/bin/env python3
"""
对比 PyTorch 和 ONNX 模型的推理结果
用于验证 ONNX 导出和推理是否正确
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import create_model, load_model_wrapper
from src.models.base import Detection
from src.utils.visualization import draw_detection_boxes


# 默认配置
COCO_VAL_PATH = "~/raw/COCO/val2017"
OUTPUT_BASE_DIR = "outputs/onnx_comparison"
CONFIDENCE_THRESHOLD = 0.25
DEFAULT_IMAGES = [
    "000000139077.jpg",  # 包含人物、电视、遥控器
    "000000139260.jpg",  # 包含蛋糕、香蕉
    "000000139871.jpg",  # 包含飞机
    "000000139872.jpg",  # 包含狗、飞盘
    "000000000285.jpg",  # 包含多人、自行车
]


def load_models(
    pt_model_path: str,
    onnx_model_path: str,
    device: str = "auto",
    conf_threshold: float = CONFIDENCE_THRESHOLD,
) -> Tuple[Any, Any, str]:
    """加载 PyTorch 和 ONNX 模型"""
    print("=" * 80)
    print("加载模型")
    print("=" * 80)

    # 加载 PyTorch 模型
    print(f"\n[PyTorch] {pt_model_path}")
    try:
        # 构建 PyTorch 权重路径
        pt_weights = None

        # 检查是否是完整路径
        if Path(pt_model_path).exists():
            pt_weights = pt_model_path
        else:
            # 尝试在 models_cache 中查找
            # 处理用户传入 yolov8m.pt 或 yolov8m 的情况
            model_name = pt_model_path.replace(".pt", "")
            pt_weights_cache = Path("models_cache") / f"{model_name}.pt"
            if pt_weights_cache.exists():
                pt_weights = str(pt_weights_cache)
            else:
                # 也尝试原始路径
                pt_weights_cache = Path("models_cache") / pt_model_path
                if pt_weights_cache.exists():
                    pt_weights = str(pt_weights_cache)

        if not pt_weights:
            print(f"  ❌ 找不到模型文件: {pt_model_path}")
            print(f"     检查路径: models_cache/{pt_model_path}")
            return None, None, ""

        # 创建并加载模型
        pt_model = create_model(
            Path(pt_weights).stem, device=device, conf_threshold=conf_threshold
        )
        load_model_wrapper(pt_model, pt_weights, Path(pt_weights).stem)
        print(f"  ✅ PyTorch 模型加载成功: {pt_model.__class__.__name__}")
        pt_info = pt_model.get_model_info()
        print(f"     类型: {pt_info.get('name', 'N/A')}")
    except Exception as e:
        print(f"  ❌ PyTorch 模型加载失败: {e}")
        import traceback

        traceback.print_exc()
        return None, None, ""

    # 加载 ONNX 模型
    print(f"\n[ONNX] {onnx_model_path}")
    try:
        # 检查 ONNX 文件是否存在
        if not Path(onnx_model_path).exists():
            print(f"  ❌ ONNX 文件不存在: {onnx_model_path}")
            return pt_model, None, ""

        # 创建 ONNX 模型（会自动加载）
        onnx_model = create_model(
            onnx_model_path, device=device, conf_threshold=conf_threshold
        )
        print(f"  ✅ ONNX 模型加载成功: {onnx_model.__class__.__name__}")
        onnx_info = onnx_model.get_model_info()
        print(f"     类型: {onnx_info.get('name', 'N/A')}")
        print(f"     格式: {onnx_info.get('format', 'N/A')}")
        print(f"     输入形状: {onnx_info.get('input_shape', 'N/A')}")
    except Exception as e:
        print(f"  ❌ ONNX 模型加载失败: {e}")
        import traceback

        traceback.print_exc()
        return pt_model, None, ""

    model_name = (
        Path(pt_model_path).stem if not pt_model_path.endswith(":onnx") else "model"
    )
    return pt_model, onnx_model, model_name


def get_class_names(model) -> Dict[int, str]:
    """获取模型的类别名称映射"""
    try:
        if hasattr(model, "model") and hasattr(model.model, "names"):
            return model.model.names
        elif hasattr(model, "names"):
            return model.names
        elif hasattr(model, "_load_class_names"):
            return model._load_class_names(Path("dummy.onnx"))
        else:
            return {}
    except Exception:
        return {}


def create_side_by_side_comparison(
    image: np.ndarray,
    pt_detections: List[Detection],
    onnx_detections: List[Detection],
    class_names: Dict[int, str],
    title: str,
) -> np.ndarray:
    """创建并排对比图"""
    h, w = image.shape[:2]

    # 绘制 PyTorch 检测结果
    pt_img = draw_detection_boxes(image, pt_detections, class_names, max_boxes=50)

    # 绘制 ONNX 检测结果
    onnx_img = draw_detection_boxes(image, onnx_detections, class_names, max_boxes=50)

    # 创建并排布局
    comparison = np.hstack([pt_img, onnx_img])

    # 添加标题栏
    title_bar = np.zeros((50, comparison.shape[1], 3), dtype=np.uint8)
    title_bar[:] = [40, 40, 40]

    # 添加标题文本
    cv2.putText(
        title_bar,
        "PyTorch Model",
        (w // 2 - 80, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        title_bar,
        "ONNX Model",
        (w + w // 2 - 60, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 150, 255),
        2,
    )

    # 添加统计信息
    stats_text = f"PT: {len(pt_detections)} detections  |  ONNX: {len(onnx_detections)} detections"
    cv2.putText(
        title_bar,
        stats_text,
        (comparison.shape[1] // 2 - 200, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    # 组合标题和对比图
    result = np.vstack([title_bar, comparison])

    return result


def process_single_image(
    pt_model,
    onnx_model,
    image_path: Path,
    output_path: Path,
    conf_threshold: float,
    model_name: str,
) -> Dict[str, Any]:
    """处理单张图片的推理和可视化"""
    result = {
        "success": False,
        "pt_detections": 0,
        "onnx_detections": 0,
        "iou_similarity": 0.0,
        "error": None,
    }

    try:
        # 读取图片
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            result["error"] = "无法读取图片"
            return result

        h, w = image_bgr.shape[:2]

        # PyTorch 推理
        print(f"  [PyTorch] 推理中...")
        pt_detections = pt_model.predict(image_bgr, conf_threshold)
        result["pt_detections"] = len(pt_detections)

        # ONNX 推理
        print(f"  [ONNX] 推理中...")
        onnx_detections = onnx_model.predict(image_bgr, conf_threshold)
        result["onnx_detections"] = len(onnx_detections)

        # 计算相似度（基于检测框数量）
        total_detections = len(pt_detections) + len(onnx_detections)
        if total_detections > 0:
            # 简单的相似度计算：检测数量相似度
            count_diff = abs(len(pt_detections) - len(onnx_detections))
            result["iou_similarity"] = max(
                0, 1 - count_diff / max(len(pt_detections), len(onnx_detections))
            )

        # 获取类别名称
        class_names = get_class_names(pt_model)

        # 创建对比图
        comparison_img = create_side_by_side_comparison(
            image_bgr,
            pt_detections,
            onnx_detections,
            class_names,
            f"{model_name} - {image_path.name}",
        )

        # 保存结果
        os.makedirs(output_path.parent, exist_ok=True)
        success = cv2.imwrite(str(output_path), comparison_img)

        if success:
            result["success"] = True
        else:
            result["error"] = "保存图片失败"

    except Exception as e:
        result["error"] = str(e)

    return result


def generate_summary_report(
    results: List[Dict[str, Any]], model_name: str, output_dir: Path
) -> None:
    """生成汇总报告"""
    print("\n" + "=" * 80)
    print("汇总报告")
    print("=" * 80)

    total_images = len(results)
    successful = sum(1 for r in results if r["success"])

    pt_total = sum(r["pt_detections"] for r in results)
    onnx_total = sum(r["onnx_detections"] for r in results)

    pt_avg = pt_total / total_images if total_images > 0 else 0
    onnx_avg = onnx_total / total_images if total_images > 0 else 0

    avg_similarity = (
        sum(r["iou_similarity"] for r in results) / total_images
        if total_images > 0
        else 0
    )

    print(f"\n模型: {model_name}")
    print(f"总图片数: {total_images}")
    print(f"成功处理: {successful}")
    print(f"\n检测统计:")
    print(f"  PyTorch 平均检测数: {pt_avg:.2f}")
    print(f"  ONNX 平均检测数: {onnx_avg:.2f}")
    print(f"  差异: {abs(pt_avg - onnx_avg):.2f}")
    print(f"\n相似度:")
    print(f"  平均相似度: {avg_similarity * 100:.1f}%")

    # 保存报告
    report_file = output_dir / "summary.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"PyTorch vs ONNX 推理对比报告\n")
        f.write(f"模型: {model_name}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"总图片数: {total_images}\n")
        f.write(f"成功处理: {successful}\n\n")

        f.write("检测统计:\n")
        f.write(f"  PyTorch 总检测数: {pt_total}\n")
        f.write(f"  PyTorch 平均检测数: {pt_avg:.2f}\n")
        f.write(f"  ONNX 总检测数: {onnx_total}\n")
        f.write(f"  ONNX 平均检测数: {onnx_avg:.2f}\n")
        f.write(f"  差异: {abs(pt_avg - onnx_avg):.2f}\n\n")

        f.write(f"平均相似度: {avg_similarity * 100:.1f}%\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("详细结果:\n")
        f.write("-" * 80 + "\n")

        for i, r in enumerate(results):
            if r["success"]:
                f.write(f"\n图片 {i + 1}:\n")
                f.write(f"  PyTorch 检测: {r['pt_detections']}\n")
                f.write(f"  ONNX 检测: {r['onnx_detections']}\n")
                f.write(f"  相似度: {r['iou_similarity'] * 100:.1f}%\n")
            else:
                f.write(f"\n图片 {i + 1}: 失败 - {r['error']}\n")

    print(f"\n✅ 报告已保存: {report_file}")


def main():
    parser = argparse.ArgumentParser(description="对比 PyTorch 和 ONNX 模型的推理结果")
    parser.add_argument(
        "--pt-model",
        type=str,
        required=True,
        help="PyTorch 模型路径或名称",
    )
    parser.add_argument(
        "--onnx-model",
        type=str,
        required=True,
        help="ONNX 模型路径",
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
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="推理设备（默认: cpu）",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PyTorch vs ONNX 推理对比工具")
    print("=" * 80)

    # 加载模型
    pt_model, onnx_model, model_name = load_models(
        args.pt_model,
        args.onnx_model,
        device=args.device,
        conf_threshold=args.conf_threshold,
    )

    if pt_model is None or onnx_model is None:
        print("\n❌ 模型加载失败，退出")
        sys.exit(1)

    # 检查数据集路径
    coco_path = Path(COCO_VAL_PATH).expanduser()
    if not coco_path.exists():
        print(f"\n❌ 数据集路径不存在: {coco_path}")
        print("   请修改脚本中的 COCO_VAL_PATH 变量")
        sys.exit(1)

    # 创建输出目录
    output_dir = Path(args.output_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n输出目录: {output_dir}")
    print(f"置信度阈值: {args.conf_threshold}")

    # 处理图片
    num_images = (
        min(len(DEFAULT_IMAGES), args.max_images)
        if args.max_images
        else len(DEFAULT_IMAGES)
    )

    print(f"\n将处理 {num_images} 张图片")
    print("=" * 80)

    results = []

    for idx in range(num_images):
        image_name = DEFAULT_IMAGES[idx]
        image_path = coco_path / image_name
        output_path = output_dir / f"comparison_{idx:02d}_{image_name}"

        print(f"\n[{idx + 1}/{num_images}] {image_name}")
        print("-" * 80)

        result = process_single_image(
            pt_model,
            onnx_model,
            image_path,
            output_path,
            args.conf_threshold,
            model_name,
        )

        results.append(result)

        if result["success"]:
            print(f"  ✅ PyTorch: {result['pt_detections']} 检测")
            print(f"  ✅ ONNX: {result['onnx_detections']} 检测")
            print(f"  ✅ 相似度: {result['iou_similarity'] * 100:.1f}%")
        else:
            print(f"  ❌ 失败: {result['error']}")

    # 生成汇总报告
    generate_summary_report(results, model_name, output_dir)

    print("\n" + "=" * 80)
    print("处理完成！")
    print(f"对比图片: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
