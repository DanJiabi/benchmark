"""CLI entry point for od-benchmark.

This module provides the main entry point for the od-benchmark CLI.
It dynamically loads all command modules and registers them with the argument parser.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for development
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="odb",
        description="Object Detection Benchmark - Performance evaluation tool",
        epilog="Run 'odb <command> --help' for more information on a command.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="odb 0.1.0",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True,
    )

    # Dynamically load all command modules
    from src.cli.commands import COMMAND_MODULES

    for cmd_module in COMMAND_MODULES:
        cmd_module.add_parser(subparsers)

    args = parser.parse_args()

    # Execute the command
    if hasattr(args, "command_func"):
        args.command_func(args)
    else:
        parser.print_help()


# Backward compatibility - export main functions
from src.cli.commands.benchmark import main as benchmark_main
from src.cli.commands.analyze import main as analyze_main


if __name__ == "__main__":
    main()
