"""Startup dependency check for the optional real-world map feature.

The grid-only mode needs just NumPy and Matplotlib, but the real-world map
mode additionally requires OSMnx and its geospatial stack. This module lets
the app offer to install the heavier packages on first run.
"""

import subprocess
import sys

# Packages required for the real-world map feature. Grid mode works without
# these; OSMnx pulls in the remaining geospatial dependencies itself.
REAL_WORLD_PACKAGES = ["osmnx", "matplotlib", "networkx", "numpy"]


def find_missing_packages(packages=REAL_WORLD_PACKAGES):
    """Return the subset of ``packages`` that cannot currently be imported."""
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def install_packages(packages):
    """Install ``packages`` with pip. Returns True on success."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
        return True
    except Exception as exc:  # noqa: BLE001 - pip can fail in many ways
        print(f"Error installing packages: {exc}")
        return False


def check_and_install_dependencies():
    """Check for missing packages and attempt to install them.

    Returns a ``(missing, installed_ok)`` tuple so a caller (e.g. the GUI)
    can decide how to inform the user. This function performs no GUI calls
    itself, keeping it usable from a headless context.
    """
    missing = find_missing_packages()
    if not missing:
        print("All required packages are already installed.")
        return [], True

    print(f"Missing packages: {', '.join(missing)}. Attempting to install...")
    installed_ok = install_packages(missing)
    if installed_ok:
        print("Required packages installed successfully.")
    return missing, installed_ok
