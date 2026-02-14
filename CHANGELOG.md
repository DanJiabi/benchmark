# Changelog

All notable changes to this project will be documented in this file.

## [0.1.1] - 2025-02-14 (Unreleased)

### Added
- Enhanced CLI with unified benchmark implementation
- Smart run_benchmark.sh script with automatic Python detection

### Changed
- **Refactored benchmark.py** (418 → 20 lines): Now a simple wrapper calling src.cli.benchmark_main()
- **Consolidated src/cli.py** (108 → 430 lines): Now contains full benchmark implementation
- **Added run_single_model()** function to src/cli.py for single model evaluation
- **Added benchmark_main()** function to src/cli.py for main benchmark logic
- **Improved run_benchmark.sh**: Now automatically detects python3/python commands

### Technical
- Code consolidation: Benchmark core logic now unified in src/cli.py
- Backward compatibility: benchmark.py remains usable for existing workflows
- Enhanced run_benchmark.sh: Automatic environment setup and command selection

### Migration Notes
Three ways to run benchmark are now available:
1. `od-benchmark benchmark [options]` (recommended)
2. `./run_benchmark.sh [options]` (automatic setup)
3. `python benchmark.py [options]` (backward compatible)

## [0.1.0] - 2025-02-14

### Added
- Initial release as a modern Python package
- Support for YOLOv8, YOLOv9, YOLOv10, RT-DETR, and Faster R-CNN models
- COCO dataset evaluation with comprehensive metrics (mAP, AR, FPS)
- Visualization support for detection boxes and performance comparisons
- CLI tool `od-benchmark` for easy command-line usage
- Public API for programmatic access to model creation and loading
- Configuration file support (YAML)
- Model weight download utility
- Comprehensive test suite

### Changed
- Refactored model classes: YOLOv8/YOLOv9/YOLOv10/RT-DETR now use unified `UltralyticsWrapper`
- Extracted common visualization functions to `src/utils/visualization.py`
- Improved project structure following modern Python packaging standards
- Added `pyproject.toml` for package configuration

### Technical
- Python 3.9+ support
- Supports CUDA, MPS (Apple Silicon), and CPU backends
- Uses setuptools build backend
- Includes development dependencies (pytest, black, ruff, mypy)

## [Unreleased]

### Planned Features
- PR curve visualization
- Confusion matrix
- Per-class mAP metrics
- Performance optimizations (multi-processing, batch processing)
- Complete unit test coverage
- API documentation (Sphinx)
- CI/CD (GitHub Actions)
- ONNX/TensorRT export support
