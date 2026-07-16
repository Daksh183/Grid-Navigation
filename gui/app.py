"""Main application window.

:class:`App` is deliberately thin: it wires together the shared
:class:`~gui.base.BaseAppMixin` with the three view mixins and owns only the
main menu. Each view's behaviour lives in its own module.
"""

import tkinter as tk
from tkinter import ttk

from gui.base import BaseAppMixin
from gui.comparison import ComparisonMixin
from gui.real_world import RealWorldMixin
from gui.utilities import UtilitiesMixin


class App(BaseAppMixin, RealWorldMixin, ComparisonMixin, UtilitiesMixin):
    """The Q-Learning Pathfinding Simulator main window."""

    def __init__(self, root):
        self.root = root
        root.title("Q-Learning Pathfinding Simulator")
        root.geometry("800x700")

        self.init_shared_state()
        self.create_main_menu()
        self.setup_hyperparameters_frame()

    def create_main_menu(self):
        # Remove everything except the persistent hyperparameter panel.
        for widget in self.root.winfo_children():
            if widget != getattr(self, "hp_frame", None):
                widget.destroy()

        self.main_menu_frame = ttk.Frame(self.root, padding="20")
        self.main_menu_frame.pack(expand=True, fill="both")

        ttk.Label(self.main_menu_frame, text="Choose a Pathfinding Mode:", font=("Arial", 16)).pack(pady=20)

        ttk.Button(self.main_menu_frame, text="Grid Pathfinding",
                   command=self.show_comparison_interface, width=30).pack(pady=10)
        ttk.Button(self.main_menu_frame, text="Real-World Map Pathfinding",
                   command=self.show_real_world_map_interface, width=30).pack(pady=10)
        ttk.Button(self.main_menu_frame, text="Utilities",
                   command=self.show_utilities_interface, width=30).pack(pady=10)


def run():
    """Create the Tk root and start the application main loop."""
    root = tk.Tk()
    App(root)
    root.mainloop()
