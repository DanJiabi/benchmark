#!/usr/bin/env python3
"""
快速开始指南 - quick_start.py

5 分钟上手 od-benchmark
"""

import subprocess
import sys


def step_1_installation():
    """步骤 1: 安装"""
    print("=" * 80)
    print("步骤 1: 安装环境")
    print("=" * 80)
    print("""
1. 创建并激活虚拟环境:
   
   mamba env create -f environment.yml --force
   conda activate benchmark

2. 以可编辑模式安装包:
   
   pip install -e .

3. 验证安装:
   
   od-benchmark --help
   
   或者运行测试脚本:
   python scripts/test_installation.py
""")


def step_2_download_weights():
    """步骤 2: 下载模型权重"""
    print("=" * 80)
    print("步骤 2: 下载模型权重")
    print("=" * 80)
    print("""
下载所有配置的模型权重:

   python scripts/download_weights.py

或者下载特定模型:

   python scripts/download_weights.py --check-only  # 只检查不下载

模型权重将下载到 models_cache/ 目录。
""")


def step_3_first_benchmark():
    """步骤 3: 运行第一个基准测试"""
    print("=" * 80)
    print("步骤 3: 运行第一个基准测试")
    print("=" * 80)
    print("""
运行简单的基准测试（使用 10 张图片）:

   od-benchmark benchmark --model yolov8n --num-images 10

这将:
- 加载 YOLOv8n 模型
- 在 10 张 COCO 验证集图片上测试
- 输出性能指标（mAP、FPS 等）
- 保存结果到 outputs/results/
""")


def step_4_visualize_results():
    """步骤 4: 可视化结果"""
    print("=" * 80)
    print("步骤 4: 可视化检测结果")
    print("=" * 80)
    print("""
查看检测框可视化:

   python examples/visualize_clean.py --model yolov8n --max-images 5

或者使用 od-benchmark:

   od-benchmark benchmark --model yolov8n --visualize --num-viz-images 10

可视化图片将保存到 outputs/visualizations/
""")


def step_5_compare_models():
    """步骤 5: 对比多个模型"""
    print("=" * 80)
    print("步骤 5: 对比多个模型")
    print("=" * 80)
    print("""
对比 YOLOv8n 和 YOLOv8s:

   od-benchmark benchmark --model yolov8n --model yolov8s --num-images 50

或者测试所有模型:

   od-benchmark benchmark --all --num-images 100

这将生成对比图表和表格。
""")


def step_6_export_model():
    """步骤 6: 导出模型"""
    print("=" * 80)
    print("步骤 6: 导出模型到 ONNX")
    print("=" * 80)
    print("""
导出 YOLOv8n 为 ONNX 格式:

   od-benchmark export --model models_cache/yolov8n.pt --format onnx

导出的模型将保存到 models_export/
""")


def step_7_analyze_custom_model():
    """步骤 7: 分析自定义模型"""
    print("=" * 80)
    print("步骤 7: 分析自定义模型")
    print("=" * 80)
    print("""
将自己的模型与基准对比:

   od-benchmark analyze \\
       --baseline yolov8n \\
       --user-model path/to/your_model.pt \\
       --num-images 50

这将生成详细的对比报告。
""")


def common_options():
    """常用选项"""
    print("=" * 80)
    print("常用命令选项")
    print("=" * 80)
    print("""
全局选项:
  --config CONFIG       指定配置文件 (默认: config.yaml)
  --device {cpu,mps,cuda}  选择推理设备

benchmark 命令:
  --model MODEL         指定要测试的模型 (可多次使用)
  --all                 测试所有配置的模型
  --num-images N        测试图片数量
  --visualize           启用可视化
  --conf-threshold N    置信度阈值
  --output-dir DIR      输出目录

analyze 命令:
  --baseline MODEL      基准模型
  --user-model MODEL    用户模型 (或权重文件路径)
  --format {json,html,csv,all}  输出格式

export 命令:
  --format {onnx,tensorrt,all}  导出格式
  --input-size H W      输入尺寸
  --dynamic             动态输入尺寸
  --simplify            简化 ONNX 模型

compare 命令:
  --formats FORMATS     对比格式 (如: pytorch,onnx)
""")


def tips():
    """提示"""
    print("=" * 80)
    print("重要提示")
    print("=" * 80)
    print("""
1. Apple Silicon (M1/M2/M3/M4) 用户:
   在运行前设置环境变量:
   export PYTORCH_ENABLE_MPS_FALLBACK=1

2. 数据集路径:
   默认使用 ~/raw/COCO/ 作为数据集路径
   可以在 config.yaml 中修改

3. 置信度阈值:
   - 0.001: 用于完整的 mAP 评估
   - 0.25: 用于可视化，结果更清晰
   - 0.5: 只显示高置信度检测结果

4. 测试图片数量:
   - 10-50: 快速测试
   - 100-500: 中等精度评估
   - 全部 (5000): 完整评估（最准确但最慢）

5. 查看帮助:
   od-benchmark --help
   od-benchmark benchmark --help
   od-benchmark analyze --help
   od-benchmark export --help
""")


def next_steps():
    """下一步"""
    print("=" * 80)
    print("下一步")
    print("=" * 80)
    print("""
现在您已经了解了基础用法，可以:

1. 查看详细示例:
   - examples/benchmark_example.py      # 基准测试示例
   - examples/analyze_example.py        # 模型对比示例
   - examples/export_example.py         # 模型导出示例
   - examples/COMPARE_PT_ONNX.md        # ONNX 对比说明

2. 阅读文档:
   - docs/ADD_CUSTOM_MODEL.md           # 添加自定义模型
   - docs/ANALYSIS_USAGE.md             # 分析功能使用
   - docs/EXPORT_GUIDE.md               # 导出指南
   - docs/FORMAT_COMPARISON.md          # 格式对比

3. 运行更多示例:
   cd examples
   python benchmark_example.py
   python analyze_example.py
   python export_example.py

4. 查看结果:
   ls outputs/results/
   ls outputs/visualizations/
""")


def run_quick_start():
    """运行快速开始指南"""
    print("\n" + "=" * 80)
    print("OD-Benchmark 快速开始指南")
    print("=" * 80)
    print("\n欢迎使用 od-benchmark！本指南将在 5 分钟内帮助您上手。\n")

    step_1_installation()
    step_2_download_weights()
    step_3_first_benchmark()
    step_4_visualize_results()
    step_5_compare_models()
    step_6_export_model()
    step_7_analyze_custom_model()
    common_options()
    tips()
    next_steps()

    print("\n" + "=" * 80)
    print("祝您使用愉快！如有问题，请查看文档或提交 Issue。")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_quick_start()
