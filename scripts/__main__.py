#!/usr/bin/env python3
"""
Script runner for od-benchmark utility scripts

Run this module directly to access utility scripts:
    python -m scripts download_weights --help
    python -m scripts test_installation
"""

import sys
from pathlib import Path


def main():
    """Main entry point for running scripts"""
    if len(sys.argv) < 2:
        print("Available scripts:")
        print("  download_weights   - Download model weights")
        print("  test_installation  - Test package installation")
        print("\nUsage: python -m scripts <script_name> [args...]")
        return 1

    script_name = sys.argv[1]
    script_args = sys.argv[2:]

    if script_name == "download_weights":
        from scripts import download_weights

        sys.argv = ["download_weights.py"] + script_args
        download_weights.main()
    elif script_name == "test_installation":
        from scripts import test_installation

        sys.argv = ["test_installation.py"] + script_args
        test_installation.main()
    else:
        print(f"Unknown script: {script_name}")
        print("\nAvailable scripts:")
        print("  download_weights   - Download model weights")
        print("  test_installation  - Test package installation")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
