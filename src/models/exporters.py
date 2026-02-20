"""
模型导出模块

支持将模型导出为 ONNX 和 TensorRT 格式
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelExporter:
    """模型导出器基类"""

    def __init__(self, model_path: str, output_dir: str = "models_export"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

    def get_output_path(self, suffix: str) -> Path:
        """获取输出文件路径"""
        model_name = self.model_path.stem
        return self.output_dir / f"{model_name}{suffix}"


class ONNXExporter(ModelExporter):
    """ONNX 模型导出器"""

    def __init__(
        self,
        model_path: str,
        output_dir: str = "models_export",
        input_size: Tuple[int, int] = (640, 640),
        opset_version: int = 12,
        simplify: bool = True,
    ):
        super().__init__(model_path, output_dir)
        self.input_size = input_size
        self.opset_version = opset_version
        self.simplify = simplify

    def export(
        self,
        dynamic: bool = False,
        batch_size: int = 1,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """
        导出模型为 ONNX 格式

        Args:
            dynamic: 是否使用动态输入尺寸
            batch_size: 批处理大小
            device: 导出设备

        Returns:
            导出结果信息
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "需要安装 ultralytics 以导出 ONNX。运行: pip install ultralytics"
            )

        logger.info(f"加载模型: {self.model_path}")
        model = YOLO(str(self.model_path))

        # 构建导出参数
        export_args = {
            "format": "onnx",
            "imgsz": self.input_size,
            "opset": self.opset_version,
            "simplify": self.simplify,
            "dynamic": dynamic,
            "batch": batch_size,
            "device": device,
        }

        logger.info(f"导出 ONNX 参数: {export_args}")

        try:
            # 执行导出
            output_path = model.export(**export_args)

            # 确保输出到指定目录
            output_file = Path(output_path)
            target_path = self.get_output_path(".onnx")

            if output_file != target_path and output_file.exists():
                import shutil

                shutil.move(str(output_file), str(target_path))
                output_file = target_path

            result = {
                "success": True,
                "format": "onnx",
                "output_path": str(output_file),
                "model_path": str(self.model_path),
                "input_size": self.input_size,
                "opset_version": self.opset_version,
                "dynamic": dynamic,
                "batch_size": batch_size,
                "file_size_mb": output_file.stat().st_size / (1024 * 1024),
            }

            logger.info(f"ONNX 导出成功: {output_file}")
            logger.info(f"文件大小: {result['file_size_mb']:.2f} MB")

            return result

        except Exception as e:
            logger.error(f"ONNX 导出失败: {e}")
            return {
                "success": False,
                "format": "onnx",
                "error": str(e),
                "model_path": str(self.model_path),
            }


class TensorRTExporter(ModelExporter):
    """TensorRT 模型导出器"""

    def __init__(
        self,
        model_path: str,
        output_dir: str = "models_export",
        input_size: Tuple[int, int] = (640, 640),
        fp16: bool = True,
        max_batch_size: int = 1,
        workspace: int = 4,
    ):
        super().__init__(model_path, output_dir)
        self.input_size = input_size
        self.fp16 = fp16
        self.max_batch_size = max_batch_size
        self.workspace = workspace

    def export(
        self,
        device: str = "0",
        half: Optional[bool] = None,
        int8: bool = False,
    ) -> Dict[str, Any]:
        """
        导出模型为 TensorRT 格式

        Args:
            device: GPU 设备编号
            half: 是否使用 FP16（默认使用初始化时的 fp16 设置）
            int8: 是否使用 INT8 量化

        Returns:
            导出结果信息
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "需要安装 ultralytics 以导出 TensorRT。运行: pip install ultralytics"
            )

        # 检查 TensorRT 环境
        self._check_tensorrt_env()

        logger.info(f"加载模型: {self.model_path}")
        model = YOLO(str(self.model_path))

        # 确定精度模式
        if half is None:
            half = self.fp16

        # 构建导出参数
        export_args = {
            "format": "engine",
            "imgsz": self.input_size,
            "half": half,
            "int8": int8,
            "batch": self.max_batch_size,
            "device": device,
            "workspace": self.workspace,
        }

        logger.info(f"导出 TensorRT 参数: {export_args}")

        try:
            # 执行导出
            output_path = model.export(**export_args)

            # 确保输出到指定目录
            output_file = Path(output_path)
            target_path = self.get_output_path(".engine")

            if output_file != target_path and output_file.exists():
                import shutil

                shutil.move(str(output_file), str(target_path))
                output_file = target_path

            result = {
                "success": True,
                "format": "tensorrt",
                "output_path": str(output_file),
                "model_path": str(self.model_path),
                "input_size": self.input_size,
                "fp16": half,
                "int8": int8,
                "max_batch_size": self.max_batch_size,
                "file_size_mb": output_file.stat().st_size / (1024 * 1024),
            }

            logger.info(f"TensorRT 导出成功: {output_file}")
            logger.info(f"文件大小: {result['file_size_mb']:.2f} MB")

            return result

        except Exception as e:
            logger.error(f"TensorRT 导出失败: {e}")
            return {
                "success": False,
                "format": "tensorrt",
                "error": str(e),
                "model_path": str(self.model_path),
            }

    def _check_tensorrt_env(self):
        """检查 TensorRT 环境"""
        try:
            import tensorrt as trt

            logger.info(f"TensorRT 版本: {trt.__version__}")
        except ImportError:
            logger.warning("未检测到 TensorRT。请安装: pip install tensorrt")
            logger.warning("注意: 导出 TensorRT 需要 CUDA 和 TensorRT 环境")


class ExportManager:
    """导出管理器 - 支持多种模型类型（YOLO、RT-DETR、Faster R-CNN）"""

    @staticmethod
    def _detect_model_type(model_path: str) -> str:
        """
        检测模型类型

        Args:
            model_path: 模型文件路径

        Returns:
            模型类型: 'ultralytics', 'faster_rcnn'
        """
        path = Path(model_path)
        stem = path.stem.lower()

        # Faster R-CNN
        if "faster" in stem and "rcnn" in stem:
            return "faster_rcnn"

        # Ultralytics 模型（YOLO 系列、RT-DETR）
        # 默认为 ultralytics（包括 RT-DETR）
        return "ultralytics"

    @staticmethod
    def export_to_onnx(
        model_path: str,
        output_dir: str = "models_export",
        input_size: Tuple[int, int] = (640, 640),
        **kwargs,
    ) -> Dict[str, Any]:
        """
        导出为 ONNX 格式的统一接口

        Args:
            model_path: 模型文件路径 (.pt)
            output_dir: 输出目录
            input_size: 输入图像尺寸
            **kwargs: 其他导出参数
                - dynamic: 动态输入尺寸
                - batch_size: 批处理大小
                - device: 导出设备
                - opset_version: ONNX opset 版本
                - simplify: 是否简化 ONNX 模型

        Returns:
            导出结果
        """
        model_type = ExportManager._detect_model_type(model_path)

        if model_type == "faster_rcnn":
            # Faster R-CNN 使用专门的导出方法
            return ExportManager._export_faster_rcnn_to_onnx(
                model_path, output_dir, input_size, **kwargs
            )
        else:
            # Ultralytics 模型（YOLO、RT-DETR）
            # RT-DETR 需要 opset 16+ (grid_sampler 操作)
            path = Path(model_path)
            is_rtdetr = "rtdetr" in path.stem.lower()

            # 确定 opset 版本
            default_opset = 16 if is_rtdetr else 12
            opset_version = kwargs.get("opset_version", default_opset)

            simplify = kwargs.get("simplify", True)
            exporter = ONNXExporter(
                model_path, output_dir, input_size, opset_version, simplify
            )

            # 提取 export() 方法需要的参数
            export_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in ["dynamic", "batch_size", "device"]
            }
            return exporter.export(**export_kwargs)

    @staticmethod
    def _export_faster_rcnn_to_onnx(
        model_path: str,
        output_dir: str = "models_export",
        input_size: Tuple[int, int] = (640, 640),
        **kwargs,
    ) -> Dict[str, Any]:
        """
        导出 Faster R-CNN 为 ONNX 格式

        Args:
            model_path: 模型文件路径（None 表示使用 torchvision 预训练）
            output_dir: 输出目录
            input_size: 输入图像尺寸

        Returns:
            导出结果
        """
        from .faster_rcnn import FasterRCNN

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        onnx_path = output_path / "faster_rcnn.onnx"

        try:
            # 创建 Faster R-CNN 模型
            model = FasterRCNN(device="cpu")

            # 加载权重
            if model_path and Path(model_path).exists():
                model.load_model(model_path)
            else:
                model.load_model(None)  # 使用 torchvision 预训练权重

            # 导出为 ONNX
            result_path = model.export_to_onnx(
                str(onnx_path),
                input_size=input_size,
                opset_version=kwargs.get("opset_version", 17),
            )

            result = {
                "success": True,
                "format": "onnx",
                "output_path": result_path,
                "model_path": model_path or "torchvision://fasterrcnn_resnet50_fpn",
                "input_size": input_size,
                "file_size_mb": Path(result_path).stat().st_size / (1024 * 1024),
            }

            logger.info(f"Faster R-CNN ONNX 导出成功: {result_path}")
            logger.info(f"文件大小: {result['file_size_mb']:.2f} MB")

            return result

        except Exception as e:
            logger.error(f"Faster R-CNN ONNX 导出失败: {e}")
            return {
                "success": False,
                "format": "onnx",
                "error": str(e),
                "model_path": model_path,
            }

    @staticmethod
    def export_to_tensorrt(
        model_path: str,
        output_dir: str = "models_export",
        input_size: Tuple[int, int] = (640, 640),
        **kwargs,
    ) -> Dict[str, Any]:
        """
        导出为 TensorRT 格式的便捷方法

        Args:
            model_path: 模型文件路径 (.pt)
            output_dir: 输出目录
            input_size: 输入图像尺寸
            **kwargs: 其他导出参数

        Returns:
            导出结果
        """
        exporter = TensorRTExporter(model_path, output_dir, input_size)
        return exporter.export(**kwargs)

    @staticmethod
    def export_all(
        model_path: str,
        output_dir: str = "models_export",
        input_size: Tuple[int, int] = (640, 640),
        formats: list = None,
    ) -> Dict[str, Any]:
        """
        导出为多种格式

        Args:
            model_path: 模型文件路径 (.pt)
            output_dir: 输出目录
            input_size: 输入图像尺寸
            formats: 要导出的格式列表 ['onnx', 'tensorrt']

        Returns:
            所有导出结果
        """
        if formats is None:
            formats = ["onnx"]

        results = {}

        for fmt in formats:
            if fmt.lower() in ["onnx", "onnxruntime"]:
                logger.info(f"\n开始导出 ONNX...")
                results["onnx"] = ExportManager.export_to_onnx(
                    model_path, output_dir, input_size
                )

            elif fmt.lower() in ["tensorrt", "trt", "engine"]:
                logger.info(f"\n开始导出 TensorRT...")
                results["tensorrt"] = ExportManager.export_to_tensorrt(
                    model_path, output_dir, input_size
                )

            else:
                logger.warning(f"不支持的导出格式: {fmt}")

        return results


def batch_export_models(
    model_paths: Optional[List[str]] = None,
    all_models: bool = False,
    model_dir: str = "models_cache",
    format: str = "onnx",
    output_dir: str = "models_export",
    input_size: Tuple[int, int] = (640, 640),
    dynamic: bool = False,
    simplify: bool = True,
    fp16: bool = True,
    int8: bool = False,
    batch_size: int = 1,
    device: str = "cpu",
    include_faster_rcnn: bool = False,
) -> Dict[str, Any]:
    """
    批量导出多个模型

    Args:
        model_paths: 模型文件路径列表
        all_models: 是否导出 models_cache 中的所有模型
        model_dir: 模型目录（当 all_models=True 时使用）
        format: 导出格式
        output_dir: 输出目录
        input_size: 输入尺寸
        dynamic: 动态输入尺寸
        simplify: 简化 ONNX
        fp16: FP16 精度
        int8: INT8 量化
        batch_size: 批处理大小
        device: 设备
        include_faster_rcnn: 是否导出 Faster R-CNN（torchvision 预训练）

    Returns:
        批量导出结果
    """
    from pathlib import Path

    # 获取模型列表
    models_to_export = []

    if all_models:
        # 从 models_cache 目录获取所有 .pt 文件
        model_dir_path = Path(model_dir)
        if model_dir_path.exists():
            models_to_export = sorted(model_dir_path.glob("*.pt"))
            models_to_export = [str(m) for m in models_to_export]
            logger.info(f"在 {model_dir} 中找到 {len(models_to_export)} 个模型")
        else:
            logger.error(f"模型目录不存在: {model_dir}")
            return {
                "success": False,
                "error": f"Model directory not found: {model_dir}",
            }

        # 添加 Faster R-CNN（如果请求）
        if include_faster_rcnn:
            models_to_export.append("faster_rcnn")
            logger.info("添加 Faster R-CNN (torchvision 预训练)")

    elif model_paths:
        models_to_export = model_paths
    else:
        logger.error("必须指定 --model 或 --all-models")
        return {"success": False, "error": "No models specified"}

    if not models_to_export:
        logger.error("没有找到要导出的模型")
        return {"success": False, "error": "No models found"}

    logger.info("=" * 70)
    logger.info("批量模型导出")
    logger.info("=" * 70)
    logger.info(f"模型数量: {len(models_to_export)}")
    logger.info(f"导出格式: {format}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"输入尺寸: {input_size}")
    logger.info("=" * 70)

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 批量导出
    all_results = []
    success_count = 0
    failed_count = 0

    for idx, model_path in enumerate(models_to_export, 1):
        logger.info(f"\n[{idx}/{len(models_to_export)}] 导出: {model_path}")
        logger.info("-" * 70)

        try:
            result = export_model_cli(
                model_path=model_path,
                format=format,
                output_dir=output_dir,
                input_size=input_size,
                dynamic=dynamic,
                simplify=simplify,
                fp16=fp16,
                int8=int8,
                batch_size=batch_size,
                device=device,
            )

            # 检查结果
            has_success = any(r.get("success") for r in result.values())
            if has_success:
                success_count += 1
            else:
                failed_count += 1

            all_results.append(
                {
                    "model": model_path,
                    "results": result,
                    "success": has_success,
                }
            )

        except Exception as e:
            logger.error(f"导出失败: {e}")
            failed_count += 1
            all_results.append(
                {
                    "model": model_path,
                    "results": {},
                    "success": False,
                    "error": str(e),
                }
            )

    # 汇总报告
    logger.info("\n" + "=" * 70)
    logger.info("批量导出完成")
    logger.info("=" * 70)
    logger.info(f"总计: {len(models_to_export)}")
    logger.info(f"成功: {success_count}")
    logger.info(f"失败: {failed_count}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 70)

    # 列出成功导出的文件
    if success_count > 0:
        logger.info("\n导出文件列表:")
        output_path = Path(output_dir)
        for ext in [".onnx", ".engine"]:
            files = list(output_path.glob(f"*{ext}"))
            for f in sorted(files):
                size_mb = f.stat().st_size / (1024 * 1024)
                logger.info(f"  - {f.name} ({size_mb:.1f} MB)")

    return {
        "success": True,
        "total": len(models_to_export),
        "success_count": success_count,
        "failed_count": failed_count,
        "output_dir": output_dir,
        "results": all_results,
    }


def export_model_cli(
    model_path: str,
    format: str = "onnx",
    output_dir: str = "models_export",
    input_size: Tuple[int, int] = (640, 640),
    dynamic: bool = False,
    simplify: bool = True,
    fp16: bool = True,
    int8: bool = False,
    batch_size: int = 1,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    CLI 导出接口 - 支持多种模型类型

    Args:
        model_path: 模型文件路径
        format: 导出格式 ('onnx', 'tensorrt', 'all')
        output_dir: 输出目录
        input_size: 输入图像尺寸
        dynamic: 动态输入尺寸（ONNX）
        simplify: 简化 ONNX 模型
        fp16: FP16 精度（TensorRT）
        int8: INT8 量化（TensorRT）
        batch_size: 批处理大小
        device: 导出设备

    Returns:
        导出结果
    """
    logger.info("=" * 60)
    logger.info("模型导出")
    logger.info("=" * 60)
    logger.info(f"模型: {model_path}")
    logger.info(f"格式: {format}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"输入尺寸: {input_size}")

    # 检测模型类型
    if model_path == "faster_rcnn":
        model_type = "faster_rcnn"
        logger.info("模型类型: Faster R-CNN (torchvision)")
    else:
        model_type = ExportManager._detect_model_type(model_path)
        logger.info(f"模型类型: {model_type}")

    if format.lower() == "all":
        formats = ["onnx"]
        # TensorRT 需要 CUDA，只在有 CUDA 时添加
        try:
            import torch

            if torch.cuda.is_available():
                formats.append("tensorrt")
                logger.info("检测到 CUDA，将同时导出 TensorRT")
        except ImportError:
            pass

        results = ExportManager.export_all(model_path, output_dir, input_size, formats)
    elif format.lower() in ["onnx", "onnxruntime"]:
        # 使用统一的导出接口（自动识别模型类型）
        # 不指定 opset_version，让 ExportManager 自动选择（RT-DETR 需要 16+）
        results = {
            "onnx": ExportManager.export_to_onnx(
                model_path,
                output_dir,
                input_size,
                dynamic=dynamic,
                batch_size=batch_size,
                device=device,
                simplify=simplify,
            )
        }
    elif format.lower() in ["tensorrt", "trt", "engine"]:
        exporter = TensorRTExporter(
            model_path, output_dir, input_size, fp16=fp16, max_batch_size=batch_size
        )
        results = {"tensorrt": exporter.export(device=device, half=fp16, int8=int8)}
    else:
        raise ValueError(f"不支持的格式: {format}")

    # 打印结果摘要
    logger.info("\n" + "=" * 60)
    logger.info("导出结果摘要")
    logger.info("=" * 60)

    success_count = 0
    for fmt, result in results.items():
        if result.get("success"):
            success_count += 1
            logger.info(f"✅ {fmt.upper()}: {result['output_path']}")
            logger.info(f"   文件大小: {result['file_size_mb']:.2f} MB")
        else:
            logger.error(f"❌ {fmt.upper()}: {result.get('error', 'Unknown error')}")

    logger.info("=" * 60)
    logger.info(f"成功: {success_count}/{len(results)}")
    logger.info("=" * 60)

    return results
