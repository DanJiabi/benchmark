"""CLI Commands for od-benchmark.

This package contains all CLI command implementations.
"""

from __future__ import annotations

from src.cli.commands import benchmark
from src.cli.commands import analyze
from src.cli.commands import export
from src.cli.commands import compare
from src.cli.commands import list
from src.cli.commands import download
from src.cli.commands import model

__all__ = [
    "benchmark",
    "analyze",
    "export",
    "compare",
    "list",
    "download",
    "model",
]

# List of all command modules for dynamic loading
COMMAND_MODULES = [
    benchmark,
    analyze,
    export,
    compare,
    list,
    download,
    model,
]
