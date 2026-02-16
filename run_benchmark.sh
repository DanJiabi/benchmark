#!/bin/bash

# Object Detection Benchmark - Run Script
# This script sets up the environment and runs the benchmark
#
# Usage:
#   ./run_benchmark.sh [options]

set -e

# Set environment variable for Apple Silicon MPS fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Check if we should use conda
if [ -n "$CONDA_DEFAULT_ENV" ]; then
	# Already in conda environment, use current python
	if command -v python3 &>/dev/null; then
		PYTHON_CMD="python3"
	elif command -v python &>/dev/null; then
		PYTHON_CMD="python"
	else
		echo "Error: Neither python3 nor python found in PATH"
		exit 1
	fi
elif command -v conda &>/dev/null; then
	# Use conda run to activate benchmark environment
	PYTHON_CMD="conda run -n benchmark python"
else
	# Use system python
	if command -v python3 &>/dev/null; then
		PYTHON_CMD="python3"
	elif command -v python &>/dev/null; then
		PYTHON_CMD="python"
	else
		echo "Error: Neither python3 nor python found in PATH"
		exit 1
	fi
fi

# Always use benchmark.py script (more reliable)
echo "Using benchmark.py script..."
$PYTHON_CMD benchmark.py "$@"
