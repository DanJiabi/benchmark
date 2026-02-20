#!/usr/bin/env python3
"""
Benchmark - Object Detection Performance Evaluation Tool

This is the primary entry point for backward compatibility.
All functionality is implemented in src.cli module.
"""

import sys
from pathlib import Path

# Ensure we're in the project directory
project_root = Path(__file__).resolve().parent
if project_root.name in ["Scripts", "scripts"]:
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

# Import and run the main function from src.cli
from src.cli import benchmark_main

if __name__ == "__main__":
    benchmark_main()
