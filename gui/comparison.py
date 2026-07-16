"""The 'Grid Pathfinding' view: run and watch any set of algorithms.

The viewer chooses everything here:

* grid size and obstacle count,
* a static or dynamic environment (and, if dynamic, whether a wall *blocks*
  the route or a *shortcut* opens, and at which episode),
* which of the four algorithms to run -- any subset, so this single screen
  covers running one algorithm or comparing several at once.

All selected algorithms are trained on the *same* grid with the *same* random
seed so the comparison is fair. Then, for every selected algorithm, its agent
is animated live on its own grid as it walks the learned path. Below the grids
the learning curves are overlaid on one plot and a metrics table is filled in
(convergence, final reward, path length, stability and training time).
"""

import random
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from rl import ALGORITHM_COLORS, ALGORITHMS, find_optimal_path
from rl.config import DEFAULT_PLANNING_STEPS
from rl.environment import generate_obstacles, has_path, make_extra_obstacles
from rl.metrics import moving_average, summarize

_METRIC_COLUMNS = [
    ("algorithm", "Algorithm", 170),
    ("convergence", "Conv. episode", 100),
    ("final_reward", "Final reward", 100),
    ("path_length", "Path length", 90),
    ("stability", "Stability (std)", 100),
    ("time", "Time (s)", 80),
    ("goal", "Goal", 55),
]

_ANIMATION_STEP_DELAY = 0.2  # seconds between agent moves


def _draw_grid(canvas, grid_size, start, end, obstacles, agent_pos=None, path=None):
    """Render a grid (with optional agent and traced path) onto a Tk canvas."""
    canvas.delete("all")
    width = canvas.winfo_width()
    height = canvas.winfo_height()
    if width <= 1 or height <= 1:
        return

    obstacles = set(obstacles)
    cell = min(width / grid_size, height / grid_size)
    offset_x = (width - grid_size * cell) / 2
    offset_y = (height - grid_size * cell) / 2

    for r in range(grid_size):
        for c in range(grid_size):
            x1 = offset_x + c * cell
            y1 = offset_y + r * cell
            color = "white"
            if (r, c) == start:
                color = "green"
            elif (r, c) == end:
                color = "red"
            elif (r, c) in obstacles:
                color = "gray"
            canvas.create_rectangle(x1, y1, x1 + cell, y1 + cell, fill=color, outline="black")

    if path:
        for (r, c) in path:
            if (r, c) != start and (r, c) != end:
                cx = offset_x + c * cell + cell / 2
                cy = offset_y + r * cell + cell / 2
                rad = cell * 0.18
                canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, fill="blue", outline="blue")

    if agent_pos:
        r, c = agent_pos
        cx = offset_x + c * cell + cell / 2
        cy = offset_y + r * cell + cell / 2
        rad = cell * 0.32
        canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad, fill="gold", outline="black")


class ComparisonMixin:
    """Grid Pathfinding UI: multi-algorithm, live animation + comparison."""

    def show_comparison_interface(self):
        self.main_menu_frame.pack_forget()

        self.comparison_frame = ttk.Frame(self.root, padding="8")
        self.comparison_frame.pack(expand=True, fill="both")

        ttk.Button(self.comparison_frame, text="< Back to Main Menu", command=self.back_to_main_menu).pack(anchor="nw")

        controls = ttk.Frame(self.comparison_frame)
        controls.pack(fill="x", pady=4)
        self._build_grid_config(controls)
        self._build_environment_config(controls)
        self._build_algorithm_config(controls)

        run_bar = ttk.Frame(self.comparison_frame)
        run_bar.pack(fill="x", pady=2)
        ttk.Button(run_bar, text="Run", command=self.run_comparison).pack(side="left", padx=5)
        self.cmp_progress_label = ttk.Label(run_bar, text="Ready.")
        self.cmp_progress_label.pack(side="left", padx=10)
        self.cmp_progress = ttk.Progressbar(run_bar, orient="horizontal", length=250, mode="determinate")
        self.cmp_progress.pack(side="right", padx=5)

        # Row of per-algorithm animated grids (populated on each run).
        self.cmp_grids_container = ttk.LabelFrame(self.comparison_frame, text="Agents searching", padding="4")
        self.cmp_grids_container.pack(fill="both", expand=True, pady=4)
        self.cmp_grid_canvases = {}

        # Combined learning-curve plot. Fixed margins (rather than an
        # auto-layout engine) avoid warnings when the canvas is briefly small.
        self.fig_cmp, self.ax_cmp = plt.subplots(figsize=(6, 2.4), dpi=100)
        self.fig_cmp.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.22)
        self.canvas_cmp = FigureCanvasTkAgg(self.fig_cmp, master=self.comparison_frame)
        self.canvas_cmp.get_tk_widget().pack(fill="x", pady=4)
        self._reset_comparison_plot()

        # Metrics table.
        table_frame = ttk.LabelFrame(self.comparison_frame, text="Metrics", padding="4")
        table_frame.pack(fill="x", pady=2)
        self.cmp_table = ttk.Treeview(
            table_frame, columns=[c[0] for c in _METRIC_COLUMNS], show="headings", height=4
        )
        for key, heading, width in _METRIC_COLUMNS:
            self.cmp_table.heading(key, text=heading)
            self.cmp_table.column(key, width=width, anchor="center")
        self.cmp_table.pack(fill="x")

    # -- control builders -----------------------------------------------------

    def _build_grid_config(self, parent):
        frame = ttk.LabelFrame(parent, text="Grid", padding="8")
        frame.pack(side="left", padx=5, fill="y")

        ttk.Label(frame, text="Size (4-12):").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self.cmp_grid_size_var = tk.IntVar(value=8)
        ttk.Spinbox(frame, from_=4, to=12, textvariable=self.cmp_grid_size_var, width=5).grid(row=0, column=1, padx=2, pady=2)

        ttk.Label(frame, text="Obstacles:").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        self.cmp_obstacles_var = tk.IntVar(value=10)
        ttk.Spinbox(frame, from_=0, to=60, textvariable=self.cmp_obstacles_var, width=5).grid(row=1, column=1, padx=2, pady=2)

    def _build_environment_config(self, parent):
        frame = ttk.LabelFrame(parent, text="Environment", padding="8")
        frame.pack(side="left", padx=5, fill="y")

        self.cmp_env_var = tk.StringVar(value="static")
        ttk.Radiobutton(frame, text="Static", variable=self.cmp_env_var, value="static",
                        command=self._toggle_dynamic_options).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(frame, text="Dynamic", variable=self.cmp_env_var, value="dynamic",
                        command=self._toggle_dynamic_options).grid(row=0, column=1, sticky="w")

        ttk.Label(frame, text="Change:").grid(row=1, column=0, sticky="w", pady=2)
        self.cmp_dyn_mode_var = tk.StringVar(value="Blocking (wall appears)")
        self.cmp_dyn_mode_combo = ttk.Combobox(
            frame, textvariable=self.cmp_dyn_mode_var, width=22, state="disabled",
            values=["Blocking (wall appears)", "Shortcut (wall opens)"],
        )
        self.cmp_dyn_mode_combo.grid(row=1, column=1, columnspan=2, sticky="w", pady=2)

        ttk.Label(frame, text="At episode:").grid(row=2, column=0, sticky="w", pady=2)
        self.cmp_change_ep_var = tk.IntVar(value=200)
        self.cmp_change_ep_entry = ttk.Entry(frame, textvariable=self.cmp_change_ep_var, width=8, state="disabled")
        self.cmp_change_ep_entry.grid(row=2, column=1, sticky="w", pady=2)

        ttk.Label(frame, text="Extra walls:").grid(row=3, column=0, sticky="w", pady=2)
        self.cmp_extra_walls_var = tk.IntVar(value=6)
        self.cmp_extra_walls_entry = ttk.Entry(frame, textvariable=self.cmp_extra_walls_var, width=8, state="disabled")
        self.cmp_extra_walls_entry.grid(row=3, column=1, sticky="w", pady=2)

    def _build_algorithm_config(self, parent):
        frame = ttk.LabelFrame(parent, text="Algorithms", padding="8")
        frame.pack(side="left", padx=5, fill="y")

        self.cmp_algo_vars = {}
        for i, name in enumerate(ALGORITHMS):
            var = tk.BooleanVar(value=True)
            self.cmp_algo_vars[name] = var
            ttk.Checkbutton(frame, text=name, variable=var).grid(row=i, column=0, sticky="w")

        ttk.Label(frame, text="Planning steps:").grid(row=len(ALGORITHMS), column=0, sticky="w", pady=(6, 0))
        self.cmp_planning_var = tk.IntVar(value=DEFAULT_PLANNING_STEPS)
        ttk.Spinbox(frame, from_=1, to=100, textvariable=self.cmp_planning_var, width=6).grid(
            row=len(ALGORITHMS) + 1, column=0, sticky="w"
        )

    def _toggle_dynamic_options(self):
        state = "normal" if self.cmp_env_var.get() == "dynamic" else "disabled"
        combo_state = "readonly" if self.cmp_env_var.get() == "dynamic" else "disabled"
        self.cmp_dyn_mode_combo.config(state=combo_state)
        self.cmp_change_ep_entry.config(state=state)
        self.cmp_extra_walls_entry.config(state=state)

    # -- plotting / table helpers --------------------------------------------

    def _reset_comparison_plot(self):
        self.ax_cmp.clear()
        self.ax_cmp.set_title("Learning Curves")
        self.ax_cmp.set_xlabel("Episode")
        self.ax_cmp.set_ylabel("Reward (smoothed)")
        self.ax_cmp.grid(True)
        self.canvas_cmp.draw()

    def _clear_metrics_table(self):
        for row in self.cmp_table.get_children():
            self.cmp_table.delete(row)

    def _setup_grid_canvases(self, selected):
        """Create one titled grid canvas per selected algorithm, side by side."""
        for child in self.cmp_grids_container.winfo_children():
            child.destroy()
        self.cmp_grid_canvases = {}

        for name in selected:
            cell = ttk.Frame(self.cmp_grids_container)
            cell.pack(side="left", expand=True, fill="both", padx=3)
            color = ALGORITHM_COLORS.get(name, "black")
            ttk.Label(cell, text=name, foreground=color, font=("Arial", 10, "bold")).pack()
            canvas = tk.Canvas(cell, bg="white", highlightthickness=1, highlightbackground="black")
            canvas.pack(expand=True, fill="both")
            self.cmp_grid_canvases[name] = canvas

    # -- run logic ------------------------------------------------------------

    def run_comparison(self):
        self.stop_running_animation()

        try:
            grid_size = self.cmp_grid_size_var.get()
            num_obstacles = self.cmp_obstacles_var.get()
            if not (4 <= grid_size <= 12):
                raise ValueError("Grid size must be between 4 and 12.")
            max_obstacles = (grid_size * grid_size) // 2
            if not (0 <= num_obstacles <= max_obstacles):
                raise ValueError(f"Obstacles must be between 0 and {max_obstacles}.")

            learning_rate, discount_factor, epsilon_decay_rate, min_epsilon, total_episodes = (
                self.read_hyperparameters()
            )

            selected = [name for name, var in self.cmp_algo_vars.items() if var.get()]
            if not selected:
                raise ValueError("Select at least one algorithm.")

            is_dynamic = self.cmp_env_var.get() == "dynamic"
            change_episode = None
            if is_dynamic:
                change_episode = self.cmp_change_ep_var.get()
                if not (0 < change_episode < total_episodes):
                    raise ValueError("Change episode must be between 1 and total episodes - 1.")
                if self.cmp_extra_walls_var.get() < 1:
                    raise ValueError("Extra walls must be at least 1 for a dynamic environment.")

            planning_steps = self.cmp_planning_var.get()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Input Error", f"Invalid input: {exc}")
            return

        start_state = (0, 0)
        end_state = (grid_size - 1, grid_size - 1)

        # Build the environment once so every algorithm sees the same world.
        base = generate_obstacles(grid_size, num_obstacles, start_state, end_state)
        if not has_path(grid_size, start_state, end_state, base):
            messagebox.showerror("Grid Error", "Could not generate a solvable grid. Reduce obstacles or enlarge the grid.")
            return

        obstacles_before = base
        obstacles_after = None
        final_obstacles = base
        if is_dynamic:
            extra = make_extra_obstacles(grid_size, base, start_state, end_state, self.cmp_extra_walls_var.get())
            if not extra:
                messagebox.showerror("Grid Error", "Could not place extra walls while keeping the maze solvable. Reduce extra walls.")
                return
            with_extra = list(set(base) | set(extra))
            if "Blocking" in self.cmp_dyn_mode_var.get():
                obstacles_before, obstacles_after = base, with_extra
            else:
                obstacles_before, obstacles_after = with_extra, base
            final_obstacles = obstacles_after

        self._reset_comparison_plot()
        self._clear_metrics_table()
        self._setup_grid_canvases(selected)
        self.cmp_progress["value"] = 0

        run_config = dict(
            grid_size=grid_size, start_state=start_state, end_state=end_state,
            obstacles_before=obstacles_before, obstacles_after=obstacles_after,
            final_obstacles=final_obstacles, change_episode=change_episode,
            total_episodes=total_episodes, learning_rate=learning_rate,
            discount_factor=discount_factor, epsilon_decay_rate=epsilon_decay_rate,
            min_epsilon=min_epsilon, planning_steps=planning_steps, selected=selected,
        )

        self.animation_thread = threading.Thread(target=self._comparison_worker, args=(run_config,), daemon=True)
        self.animation_thread.start()

    def _comparison_worker(self, cfg):
        results = {}  # name -> dict(rewards, metrics, path)
        n = len(cfg["selected"])

        for i, name in enumerate(cfg["selected"], start=1):
            if self.animation_stop_event.is_set():
                return
            self.root.after(0, lambda nm=name, idx=i: self.cmp_progress_label.config(text=f"Training {nm} ({idx}/{n})..."))

            spec = ALGORITHMS[name]
            kwargs = dict(
                total_episodes=cfg["total_episodes"], learning_rate=cfg["learning_rate"],
                discount_factor=cfg["discount_factor"], epsilon_decay_rate=cfg["epsilon_decay_rate"],
                min_epsilon=cfg["min_epsilon"], change_episode=cfg["change_episode"],
                obstacles_after=cfg["obstacles_after"],
                progress_callback=lambda cur, tot: self.root.after(0, lambda: self.cmp_progress.config(value=cur / tot * 100)),
            )
            if "planning_steps" in spec["supports"]:
                kwargs["planning_steps"] = cfg["planning_steps"]

            # Same seed for every algorithm => fair comparison.
            random.seed(0)
            np.random.seed(0)

            t0 = time.time()
            q_table, episode_rewards = spec["fn"](
                cfg["grid_size"], cfg["start_state"], cfg["end_state"], cfg["obstacles_before"], **kwargs
            )
            train_time = time.time() - t0

            if q_table is None:
                results[name] = {"rewards": [], "metrics": None, "path": []}
                continue

            path = find_optimal_path(
                q_table, cfg["grid_size"], cfg["start_state"], cfg["end_state"], cfg["final_obstacles"]
            )
            metrics = summarize(episode_rewards, path=path, end_state=cfg["end_state"], train_time=train_time)
            results[name] = {"rewards": episode_rewards, "metrics": metrics, "path": path}

        self.root.after(0, lambda: self._show_comparison_results(results, cfg))

    def _show_comparison_results(self, results, cfg):
        # --- Overlaid learning curves --------------------------------------
        self.ax_cmp.clear()
        title = "Learning Curves"
        if cfg["change_episode"]:
            title += f"  (env changes @ ep {cfg['change_episode']})"
        self.ax_cmp.set_title(title)
        self.ax_cmp.set_xlabel("Episode")
        self.ax_cmp.set_ylabel("Reward (smoothed)")
        self.ax_cmp.grid(True)

        window = 10
        for name, data in results.items():
            rewards = data["rewards"]
            if not rewards:
                continue
            smoothed = moving_average(rewards, window=window)
            x = range(window - 1, window - 1 + len(smoothed))
            self.ax_cmp.plot(x, smoothed, label=name, color=ALGORITHM_COLORS.get(name))

        if cfg["change_episode"]:
            self.ax_cmp.axvline(cfg["change_episode"], color="gray", linestyle="--", linewidth=1, label="env change")
        self.ax_cmp.legend(fontsize=8)
        self.canvas_cmp.draw()

        # --- Metrics table -------------------------------------------------
        self._clear_metrics_table()
        for name, data in results.items():
            metrics = data["metrics"]
            if metrics is None:
                self.cmp_table.insert("", "end", values=(name, "failed", "-", "-", "-", "-", "-"))
                continue
            self.cmp_table.insert("", "end", values=(
                name,
                metrics["convergence_episode"],
                f"{metrics['final_reward']:.1f}",
                metrics["path_length"] if metrics["path_length"] is not None else "-",
                f"{metrics['stability']:.1f}",
                f"{metrics['train_time']:.2f}",
                "yes" if metrics["reached_goal"] else "no",
            ))

        # --- Live agent animation on each grid -----------------------------
        self.cmp_progress_label.config(text="Animating agents...")
        self.animation_thread = threading.Thread(target=self._animate_grids, args=(results, cfg), daemon=True)
        self.animation_thread.start()

    def _animate_grids(self, results, cfg):
        grid_size = cfg["grid_size"]
        start = cfg["start_state"]
        end = cfg["end_state"]
        obstacles = cfg["final_obstacles"]

        paths = {name: data["path"] for name, data in results.items() if data["path"]}
        if not paths:
            self.root.after(0, lambda: self.cmp_progress_label.config(text="Done (no path found)."))
            return

        # Draw the initial grids (agent at start).
        for name, canvas in self.cmp_grid_canvases.items():
            self.root.after(0, lambda cv=canvas, p=paths.get(name): _draw_grid(
                cv, grid_size, start, end, obstacles, agent_pos=start if p else None))
        time.sleep(0.3)

        max_len = max(len(p) for p in paths.values())
        for step in range(max_len):
            if self.animation_stop_event.is_set():
                return
            for name, canvas in self.cmp_grid_canvases.items():
                path = paths.get(name)
                if not path:
                    continue
                pos = path[min(step, len(path) - 1)]
                self.root.after(0, lambda cv=canvas, p=pos: _draw_grid(
                    cv, grid_size, start, end, obstacles, agent_pos=p))
            time.sleep(_ANIMATION_STEP_DELAY)

        if self.animation_stop_event.is_set():
            return

        # Final frame: show the full traced path on each grid.
        for name, canvas in self.cmp_grid_canvases.items():
            path = paths.get(name)
            self.root.after(0, lambda cv=canvas, p=path: _draw_grid(
                cv, grid_size, start, end, obstacles, path=p))
        self.root.after(0, lambda: self.cmp_progress_label.config(text="Done."))
