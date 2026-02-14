#!/bin/bash

# Object Detection Benchmark - Run Script
# This script sets up the environment and runs the benchmark
#
# Usage:
#   ./run_benchmark.sh [options]
#   od-benchmark benchmark [options]
#   python benchmark.py [options]

set -e

# Set environment variable for Apple Silicon MPS fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Determine which Python to use
if command -v python3 &>/dev/null; then
	PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
	PYTHON_CMD="python"
else
	echo "Error: Neither python3 nor python found in PATH"
	exit 1
fi

# Determine which command to use
if command -v od-benchmark &>/dev/null; then
	# Use CLI tool
	echo "Using od-benchmark CLI tool..."
	od-benchmark benchmark "$@"
else
	# Use Python script
	echo "Using benchmark.py script..."
	$PYTHON_CMD benchmark.py "$@"
fi
