import json
import time
from typing import Dict, List, Any
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from ..models.base import Detection


class COCOMetrics:
    def __init__(self, annotations_file: str):
        self.coco_gt = COCO(annotations_file)
        self.image_ids = self.coco_gt.getImgIds()

    def compute_metrics(
        self, predictions: List[Dict[str, Any]], iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        if not predictions:
            return {
                "AP@0.50": 0.0,
                "AP@0.50:0.95": 0.0,
                "AP@0.75": 0.0,
                "AP_small": 0.0,
                "AP_medium": 0.0,
                "AP_large": 0.0,
                "AR1": 0.0,
                "AR10": 0.0,
                "AR100": 0.0,
                "AR_small": 0.0,
                "AR_medium": 0.0,
                "AR_large": 0.0,
            }

        coco_dt = self.coco_gt.loadRes(predictions)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")

        # 重要修复: 只评估有预测的图片
        # 这避免了 --num-images 参数导致的 mAP 被低估问题
        evaluated_img_ids = list(set(p["image_id"] for p in predictions))
        coco_eval.params.imgIds = evaluated_img_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {
            "AP@0.50": coco_eval.stats[0],
            "AP@0.50:0.95": coco_eval.stats[1],
            "AP@0.75": coco_eval.stats[2],
            "AP_small": coco_eval.stats[3],
            "AP_medium": coco_eval.stats[4],
            "AP_large": coco_eval.stats[5],
            "AR1": coco_eval.stats[6],
            "AR10": coco_eval.stats[7],
            "AR100": coco_eval.stats[8],
            "AR_small": coco_eval.stats[9],
            "AR_medium": coco_eval.stats[10],
            "AR_large": coco_eval.stats[11],
        }

        return metrics

    def predictions_to_coco_format(
        self, all_detections: Dict[int, List[Detection]]
    ) -> List[Dict[str, Any]]:
        predictions = []
        ann_id = 1

        for image_id, detections in all_detections.items():
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                width = x2 - x1
                height = y2 - y1

                pred = {
                    "image_id": image_id,
                    "category_id": det.class_id,
                    "bbox": [x1, y1, width, height],
                    "score": det.confidence,
                    "id": ann_id,
                }
                predictions.append(pred)
                ann_id += 1

        return predictions


class PerformanceMetrics:
    def __init__(self):
        self.inference_times = []
        self.total_images = 0

    def start_timer(self) -> float:
        return time.time()

    def end_timer(self, start_time: float) -> float:
        return time.time() - start_time

    def add_inference_time(self, inference_time: float) -> None:
        self.inference_times.append(inference_time)
        self.total_images += 1

    def compute_performance_stats(self) -> Dict[str, float]:
        if not self.inference_times:
            return {}

        times = np.array(self.inference_times)
        avg_time = np.mean(times)
        fps = 1.0 / avg_time if avg_time > 0 else 0

        return {
            "avg_inference_time_ms": avg_time * 1000,
            "min_inference_time_ms": np.min(times) * 1000,
            "max_inference_time_ms": np.max(times) * 1000,
            "std_inference_time_ms": np.std(times) * 1000,
            "fps": fps,
            "total_images": self.total_images,
        }


class MetricsAggregator:
    def __init__(self):
        self.model_results = defaultdict(dict)

    def add_model_result(
        self,
        model_name: str,
        coco_metrics: Dict[str, float],
        performance: Dict[str, float],
        model_info: Dict[str, Any],
    ) -> None:
        self.model_results[model_name].update(
            {
                "coco_metrics": coco_metrics,
                "performance": performance,
                "model_info": model_info,
            }
        )

    def get_model_result(self, model_name: str) -> Dict[str, Any]:
        return self.model_results.get(model_name, {})

    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.model_results)

    def compare_models(self) -> Dict[str, Any]:
        if not self.model_results:
            return {}

        comparison = {
            "model_names": list(self.model_results.keys()),
            "mAP50": [],
            "mAP50-95": [],
            "fps": [],
            "params": [],
        }

        for name, result in self.model_results.items():
            comparison["mAP50"].append(result["coco_metrics"]["mAP50"])
            comparison["mAP50-95"].append(result["coco_metrics"]["mAP50-95"])
            comparison["fps"].append(result["performance"]["fps"])
            comparison["params"].append(result["model_info"].get("params", 0))

        return comparison

    def save_results(self, output_path: str) -> None:
        with open(output_path, "w") as f:
            json.dump(dict(self.model_results), f, indent=2)
