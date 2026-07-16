"""General-purpose helpers: dependency management and text utilities."""

from .dependencies import check_and_install_dependencies, find_missing_packages
from .text_tools import (
    get_working_directory,
    remove_dirty_chars,
    simulate_file_upload,
)

__all__ = [
    "check_and_install_dependencies",
    "find_missing_packages",
    "get_working_directory",
    "remove_dirty_chars",
    "simulate_file_upload",
]
