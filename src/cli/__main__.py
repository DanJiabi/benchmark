"""CLI entry point for direct module execution."""

# Set PYTORCH_ENABLE_MPS_FALLBACK for Apple Silicon compatibility
# This MUST be done before ANY other imports
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from src.cli import main

if __name__ == "__main__":
    main()
