from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_metrics_comparison(
    results: Dict[str, Dict[str, Any]],
    metrics: List[str],
    output_path: str = None,
    figsize: tuple = (12, 6),
) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        model_names = list(results.keys())
        values = []

        for name in model_names:
            if metric in results[name]["coco_metrics"]:
                values.append(results[name]["coco_metrics"][metric])
            elif metric in results[name]["performance"]:
                values.append(results[name]["performance"][metric])
            else:
                values.append(0)

        ax = axes[idx]
        bars = ax.bar(model_names, values)
        ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=45)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_fps_vs_map(
    results: Dict[str, Dict[str, Any]],
    output_path: str = None,
    figsize: tuple = (10, 8),
) -> None:
    model_names = []
    fps_values = []
    map_values = []

    for name, result in results.items():
        model_names.append(name)
        fps_values.append(result["performance"]["fps"])
        map_values.append(result["coco_metrics"]["mAP50"])

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(fps_values, map_values, s=200, alpha=0.7)

    for i, name in enumerate(model_names):
        ax.annotate(
            name,
            (fps_values[i], map_values[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("FPS (Frames per Second)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("FPS vs mAP@0.5 Comparison")
    ax.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def plot_model_size_vs_performance(
    results: Dict[str, Dict[str, Any]],
    output_path: str = None,
    figsize: tuple = (10, 8),
) -> None:
    model_names = []
    params = []
    map_values = []
    sizes_mb = []

    for name, result in results.items():
        model_names.append(name)
        params.append(result["model_info"].get("params", 0))
        map_values.append(result["coco_metrics"]["mAP50"])
        sizes_mb.append(result["model_info"].get("model_size_mb", params[-1] * 4))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.scatter(params, map_values, s=200, alpha=0.7)
    for i, name in enumerate(model_names):
        ax1.annotate(
            name,
            (params[i], map_values[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax1.set_xlabel("Parameters (Millions)")
    ax1.set_ylabel("mAP@0.5")
    ax1.set_title("Model Size vs Accuracy")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(sizes_mb, map_values, s=200, alpha=0.7)
    for i, name in enumerate(model_names):
        ax2.annotate(
            name,
            (sizes_mb[i], map_values[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax2.set_xlabel("Model Size (MB)")
    ax2.set_ylabel("mAP@0.5")
    ax2.set_title("File Size vs Accuracy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def generate_results_table(
    results: Dict[str, Dict[str, Any]], metrics: List[str] = None
) -> pd.DataFrame:
    if metrics is None:
        metrics = ["mAP50", "mAP50-95", "fps", "params"]

    table_data = []

    for name, result in results.items():
        row = {"Model": name}

        for metric in metrics:
            if metric in result["coco_metrics"]:
                row[metric] = f"{result['coco_metrics'][metric]:.4f}"
            elif metric in result["performance"]:
                row[metric] = f"{result['performance'][metric]:.2f}"
            elif metric == "params" and "model_info" in result:
                row[metric] = f"{result['model_info'].get(metric, 0):.2f}M"
            else:
                row[metric] = "N/A"

        table_data.append(row)

    df = pd.DataFrame(table_data)
    return df.set_index("Model")


def save_results_table(
    results: Dict[str, Dict[str, Any]], output_path: str, format: str = "csv"
) -> None:
    df = generate_results_table(results)

    if format == "csv":
        df.to_csv(output_path)
    elif format == "markdown":
        df.to_markdown(output_path)
    elif format == "latex":
        df.to_latex(output_path)
