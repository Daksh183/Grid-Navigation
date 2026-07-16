"""Shared GUI state and widgets common to every view.

:class:`BaseAppMixin` holds the pieces that the grid, real-world and utility
views all rely on: the shared training state, the always-visible
hyperparameter panel, progress reporting, learning-curve plotting and the
Q-table popup. The concrete :class:`gui.app.App` combines this with the
per-view mixins.
"""

import threading
import tkinter as tk
from tkinter import ttk

import matplotlib.pyplot as plt

from rl.config import (
    ACTIONS,
    DEFAULT_DISCOUNT_FACTOR,
    DEFAULT_EPSILON_DECAY_RATE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIN_EPSILON,
    DEFAULT_TOTAL_EPISODES,
)


class BaseAppMixin:
    """State and widgets shared across all views."""

    def init_shared_state(self):
        """Initialise the training state shared by every view."""
        self.current_q_table = None
        self.current_grid_params = None  # (grid_size, start, end, obstacles)
        self.current_tab_trained = None  # "original" or "real_world"

        # Real-world map specific state.
        self.current_osm_graph = None
        self.real_world_grid_to_node_map = None
        self.real_world_node_to_grid_map = None
        self.real_world_min_coords = None
        self.real_world_max_coords = None
        self.real_world_graph_start_node = None
        self.real_world_graph_end_node = None

        # Background training/animation.
        self.animation_thread = None
        self.animation_stop_event = threading.Event()

    def setup_hyperparameters_frame(self):
        """Build the hyperparameter panel pinned to the bottom of the window."""
        self.hp_frame = ttk.LabelFrame(self.root, text="Hyperparameters", padding="10")
        self.hp_frame.pack(side="bottom", fill="x", pady=5, padx=10)

        ttk.Label(self.hp_frame, text="Learning Rate:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.lr_var = tk.DoubleVar(value=DEFAULT_LEARNING_RATE)
        ttk.Entry(self.hp_frame, textvariable=self.lr_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(self.hp_frame, text="Discount Factor:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
        self.df_var = tk.DoubleVar(value=DEFAULT_DISCOUNT_FACTOR)
        ttk.Entry(self.hp_frame, textvariable=self.df_var, width=10).grid(row=0, column=3, padx=5, pady=2)

        ttk.Label(self.hp_frame, text="Epsilon Decay Rate:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.edr_var = tk.DoubleVar(value=DEFAULT_EPSILON_DECAY_RATE)
        ttk.Entry(self.hp_frame, textvariable=self.edr_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(self.hp_frame, text="Min Epsilon:").grid(row=1, column=2, sticky="w", padx=5, pady=2)
        self.min_eps_var = tk.DoubleVar(value=DEFAULT_MIN_EPSILON)
        ttk.Entry(self.hp_frame, textvariable=self.min_eps_var, width=10).grid(row=1, column=3, padx=5, pady=2)

        ttk.Label(self.hp_frame, text="Total Episodes:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.episodes_var = tk.IntVar(value=DEFAULT_TOTAL_EPISODES)
        ttk.Entry(self.hp_frame, textvariable=self.episodes_var, width=10).grid(row=2, column=1, padx=5, pady=2)

        for i in range(4):
            self.hp_frame.grid_columnconfigure(i, weight=1)

    def read_hyperparameters(self):
        """Read and validate the hyperparameters. Raises ``ValueError`` if invalid."""
        learning_rate = self.lr_var.get()
        discount_factor = self.df_var.get()
        epsilon_decay_rate = self.edr_var.get()
        min_epsilon = self.min_eps_var.get()
        total_episodes = self.episodes_var.get()

        if not (
            0 < learning_rate <= 1
            and 0 <= discount_factor <= 1
            and 0 < epsilon_decay_rate <= 1
            and 0 <= min_epsilon < 1
            and total_episodes > 0
        ):
            raise ValueError("Hyperparameters out of valid range.")

        return learning_rate, discount_factor, epsilon_decay_rate, min_epsilon, total_episodes

    def stop_running_animation(self, timeout=0.2):
        """Signal any running training/animation thread to stop and join it."""
        self.animation_stop_event.set()
        if self.animation_thread and self.animation_thread.is_alive():
            self.animation_thread.join(timeout=timeout)
        self.animation_stop_event.clear()

    def update_progress(self, current_episode, total_episodes, tab_name="original"):
        """Update the progress bar for the given view."""
        progress = (current_episode / total_episodes) * 100
        if tab_name == "original":
            self.progress_og["value"] = progress
            self.progress_label_og["text"] = f"Training Progress: {progress:.1f}%"
        elif tab_name == "real_world":
            self.progress_rw["value"] = progress
            self.progress_label_rw["text"] = f"Training Progress: {progress:.1f}%"
        self.root.update_idletasks()

    def plot_learning_curve(self, episode_rewards, ax, canvas, title):
        """Draw the total-reward-per-episode learning curve."""
        ax.clear()
        if episode_rewards:
            ax.plot(episode_rewards)
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Reward")
        else:
            ax.text(
                0.5,
                0.5,
                "No episode rewards to plot",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
        ax.set_title(f"{title} Learning Curve")
        ax.grid(True)
        canvas.draw()

    def show_q_table_popup(self, mode):
        """Open a scrollable popup that prints the current Q-table."""
        from tkinter import messagebox

        if self.current_q_table is None:
            messagebox.showwarning(
                "No Q-Table", "No Q-Table available. Please run a training simulation first."
            )
            return

        popup = tk.Toplevel(self.root)
        popup.title(f"{mode.replace('_', ' ').title()} Q-Table")
        popup.geometry("600x400")

        text_frame = ttk.Frame(popup)
        text_frame.pack(expand=True, fill="both", padx=10, pady=10)

        text_widget = tk.Text(text_frame, wrap="none", font=("Consolas", 10))
        text_widget.pack(side="left", expand=True, fill="both")

        vsb = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        vsb.pack(side="right", fill="y")
        text_widget.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(popup, orient="horizontal", command=text_widget.xview)
        hsb.pack(side="bottom", fill="x")
        text_widget.configure(xscrollcommand=hsb.set)

        q_table_str = "Q-Table:\n"
        grid_size = self.current_q_table.shape[0]
        for r in range(grid_size):
            for c in range(grid_size):
                q_table_str += f"\nState ({r}, {c}):\n"
                for action_idx, q_value in enumerate(self.current_q_table[r, c]):
                    q_table_str += f"  Action {action_idx} {ACTIONS[action_idx]}: {q_value:.4f}\n"
            q_table_str += "-" * 50 + "\n"

        text_widget.insert(tk.END, q_table_str)
        text_widget.config(state="disabled")

    def back_to_main_menu(self):
        """Tear down the current view and return to the main menu."""
        for attr in ("comparison_frame", "real_world_frame", "utilities_frame"):
            frame = getattr(self, attr, None)
            if frame is not None and frame.winfo_exists():
                frame.destroy()

        # Stop any ongoing animation threads.
        self.animation_stop_event.set()
        if self.animation_thread and self.animation_thread.is_alive():
            self.animation_thread.join(timeout=0.5)
        self.animation_stop_event.clear()

        self.init_shared_state()
        plt.close("all")
        self.create_main_menu()
