"""Entry point for the Q-Learning Pathfinding Simulator.

Run with::

    python main.py

The grid-only mode needs just NumPy and Matplotlib. The real-world map mode
additionally requires OSMnx; if it is missing, the app still launches and
only warns when you open that view.
"""

from utils.dependencies import find_missing_packages


def main():
    missing = find_missing_packages()
    if missing:
        print(
            "Note: the following packages are missing and are only needed for "
            f"the Real-World Map mode: {', '.join(missing)}\n"
            "Install them with:  pip install -r requirements.txt\n"
            "(Grid mode works without them.)\n"
        )

    # Imported here so the app can start even before optional deps are present.
    from gui.app import run

    run()


if __name__ == "__main__":
    main()
