"""Tkinter GUI, split into one module per view plus a shared base.

The public entry points are :class:`gui.app.App` and :func:`gui.app.run`.
"""

from gui.app import App, run

__all__ = ["App", "run"]
