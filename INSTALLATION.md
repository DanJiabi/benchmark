# Installation Guide

## Traditional Installation

You can still use the traditional way to install and run this project:

```bash
# Create virtual environment
mamba env create -f environment.yml --force

# Activate environment
conda activate benchmark

# Run benchmarks
python benchmark.py --model yolov8n
```

## Modern Python Package Installation

You can now install this project as a modern Python package:

### 1. Install in Editable Mode (Recommended for Development)

```bash
# Create virtual environment
mamba env create -f environment.yml --force

# Activate environment
conda activate benchmark

# Install package in editable mode
pip install -e .
```

After installation, you can use the package in two ways:

#### Using the CLI Tool

```bash
# Show help
od-benchmark --help

# Run benchmark
od-benchmark benchmark --model yolov8n --num-images 10

# Run all models
od-benchmark benchmark --all --num-images 100

# Run with visualization
od-benchmark benchmark --model yolov8n --visualize --num-viz-images 20
```

#### Using Python API

```python
from src.models import create_model, load_model_wrapper

# Create a model
model = create_model("yolov8n", device="auto", conf_threshold=0.25)

# Load weights
load_model_wrapper(model, "models_cache/yolov8n.pt", "YOLOv8n")

# Use the model
detections = model.predict(image)
```

### 2. Install from Local Source

```bash
# Build and install
pip install .
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This will install additional development tools:
- pytest (testing)
- pytest-cov (coverage)
- mypy (type checking)
- black (code formatting)
- ruff (linting)

## Package Structure

After installation, the package provides the following components:

### Public API

```python
import src

# Model factory
src.create_model(model_name, device, conf_threshold)
src.load_model_wrapper(model, weights_path, model_name)

# Base classes
src.BaseModel
src.Detection
```

### CLI Commands

```bash
od-benchmark                    # Main CLI
od-benchmark benchmark           # Run benchmark
od-benchmark --version          # Show version
```

### Traditional Scripts

You can still use the traditional Python scripts directly:

```bash
python benchmark.py --model yolov8n
python examples/visualize_clean.py --model yolov8n
python download_weights.py
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py
```

### Code Quality

```bash
# Format code
black src/ tests/ examples/

# Lint code
ruff check src/

# Type check
mypy src/
```

### Building Distribution

```bash
# Build wheel and source distribution
python -m build

# Install twine if not already installed
pip install twine

# Upload to PyPI (if you have permissions)
twine upload dist/*
```

## Package Metadata

- **Name**: od-benchmark
- **Version**: 0.1.0
- **License**: MIT
- **Python Version**: 3.9+
- **Author**: Contributors

## Troubleshooting

### Installation Issues

If you encounter issues during installation:

1. **Make sure you have the latest pip**:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **Check Python version**:
   ```bash
   python --version
   # Should be 3.9 or higher
   ```

3. **Clean and reinstall**:
   ```bash
   pip uninstall od-benchmark
   rm -rf build dist *.egg-info
   pip install -e .
   ```

### Import Issues

If you cannot import the package:

1. **Check installation**:
   ```bash
   pip list | grep od-benchmark
   ```

2. **Verify editable installation**:
   ```bash
   pip show od-benchmark
   # Should show "Editable: true" and point to your project directory
   ```

3. **Reinstall**:
   ```bash
   pip uninstall -y od-benchmark
   pip install -e .
   ```

## Quick Start Examples

### Example 1: Quick Test with YOLOv8n

```bash
# Traditional way
python benchmark.py --model yolov8n --num-images 10

# Using CLI tool
od-benchmark benchmark --model yolov8n --num-images 10
```

### Example 2: Compare Multiple Models

```bash
# Traditional way
python benchmark.py --model yolov8n --model yolov8s --model yolov8m

# Using CLI tool
od-benchmark benchmark --model yolov8n --model yolov8s --model yolov8m
```

### Example 3: Full Benchmark with Visualization

```bash
# Traditional way
python benchmark.py --all --visualize --num-viz-images 50

# Using CLI tool
od-benchmark benchmark --all --visualize --num-viz-images 50
```

## Migration from Traditional Installation

If you were using the project before it was converted to a package:

1. **No changes needed** - All traditional scripts still work:
   - `benchmark.py`
   - `examples/visualize_clean.py`
   - `download_weights.py`

2. **Optional: Use CLI tool** - For a more modern experience:
   ```bash
   od-benchmark benchmark --model yolov8n
   ```

3. **Optional: Import as package** - For Python API usage:
   ```python
   from src.models import create_model
   model = create_model("yolov8n")
   ```

## Next Steps

After successful installation:

1. **Download model weights**:
   ```bash
   python scripts/download_weights.py
   ```

2. **Verify installation**:
   ```bash
   python scripts/test_installation.py
   ```

3. **Run your first benchmark**:
   
   **Option 1: Using CLI tool (recommended)**:
   ```bash
   od-benchmark benchmark --model yolov8n --num-images 100
   ```

   **Option 2: Using wrapper script**:
   ```bash
   ./run_benchmark.sh --model yolov8n --num-images 100
   ```

   **Option 3: Using Python script**:
   ```bash
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   python benchmark.py --model yolov8n --num-images 100
   ```

For more information, see the main [README.md](README.md).
