# Scripts Directory

This directory contains utility scripts for the od-benchmark package.

## Available Scripts

### 1. download_weights.py

Download model weights from URLs configured in config.yaml.

#### Usage

```bash
# Basic usage - download all models from config.yaml
python scripts/download_weights.py

# Use specific config file
python scripts/download_weights.py --config config_test.yaml

# Specify cache directory
python scripts/download_weights.py --cache-dir /path/to/cache

# Overwrite existing files
python scripts/download_weights.py --overwrite

# Check file integrity only (no download)
python scripts/download_weights.py --check-only
```

#### Features

- Batch download all configured model weights
- Check file integrity (size, PyTorch model loading)
- Automatic re-download of incomplete files
- Progress display
- Save download results to file

#### File Integrity Check

- File size validation (prevents small error files)
- PyTorch model loading verification
- File readability test

---

### 2. test_installation.py

Verify that the od-benchmark package is correctly installed.

#### Usage

```bash
# Run all tests
python scripts/test_installation.py
```

#### What It Tests

1. **Module Imports** - All modules can be imported correctly
2. **Model Creation** - Both YOLO and FasterRCNN models can be created
3. **CLI** - Command line interface is available

#### Expected Output

```
============================================================
OD-Benchmark Installation Test
============================================================

============================================================
Testing Imports
============================================================
✅ Imported BaseModel and Detection from src.models.base
✅ Imported create_model and load_model_wrapper from src.models
✅ Imported UltralyticsWrapper from src.models.ultralytics_wrapper
✅ Imported FasterRCNN from src.models.faster_rcnn
✅ Imported visualization functions from src.utils.visualization

============================================================
Testing Model Creation
============================================================
✅ Created YOLO model: UltralyticsWrapper
✅ Created FasterRCNN model: FasterRCNN

============================================================
Testing CLI
============================================================
✅ od-benchmark --help works
✅ od-benchmark benchmark --help works

============================================================
Test Summary
============================================================
Imports              ✅ PASS
Model Creation       ✅ PASS
CLI                  ✅ PASS
============================================================

✅ All tests passed! Installation successful.
```

---

## Running Scripts

All scripts in this directory can be run independently:

```bash
# Run script directly
python scripts/download_weights.py --help
python scripts/test_installation.py

# Or use Python module syntax
python -m scripts download_weights --help
python -m scripts test_installation
```

---

## Script Requirements

All scripts require the od-benchmark package to be installed:

```bash
# Install in editable mode
pip install -e .
```

Or have the project root in your Python path:

```bash
# Add to path when running
python scripts/download_weights.py
```

---

## Development

When adding new scripts to this directory:

1. Make the script executable (optional):
   ```bash
   chmod +x scripts/your_script.py
   ```

2. Add a shebang line:
   ```python
   #!/usr/bin/env python3
   ```

3. Follow the same style as existing scripts:
   - Use argparse for command-line arguments
   - Include help text
   - Add docstrings

4. Update `__main__.py` if you want to run via:
   ```bash
   python -m scripts your_script
   ```

---

## See Also

- [Main README](../README.md) - Complete documentation
- [Installation Guide](../INSTALLATION.md) - Installation instructions
- [Changelog](../CHANGELOG.md) - Version history
