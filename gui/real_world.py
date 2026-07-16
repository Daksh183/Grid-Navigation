"""The 'Real-World Map' pathfinding view.

A street network is downloaded with OSMnx, discretised into a grid, trained
with Q-Learning, and the resulting route is animated on top of the real map.
OSMnx and Matplotlib's OSMnx plotting are imported lazily so this view only
pulls in the geospatial stack when it is actually used.
"""

import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from rl.config import REAL_WORLD_GRID_SIZE
from rl.q_learning import find_optimal_path, train_q_learning
from realworld.grid import graph_to_simplified_grid
from realworld.osm import get_real_world_graph, get_start_end_nodes


class RealWorldMixin:
    """Real-world-map UI, training and animation."""

    def show_real_world_map_interface(self):
        self.main_menu_frame.pack_forget()

        self.real_world_frame = ttk.Frame(self.root, padding="10")
        self.real_world_frame.pack(expand=True, fill="both")

        control_frame_rw = ttk.Frame(self.real_world_frame)
        control_frame_rw.pack(pady=5, fill="x")
        ttk.Button(control_frame_rw, text="< Back to Main Menu", command=self.back_to_main_menu).pack(side="left", padx=5)

        place_frame = ttk.Frame(control_frame_rw)
        place_frame.pack(side="left", padx=10, fill="x", expand=True)

        ttk.Label(place_frame, text="Place Name (e.g., 'Berkeley, CA' or 'Random Location'):").grid(
            row=0, column=0, sticky="w", padx=5
        )
        self.place_name_var = tk.StringVar(value="San Francisco, CA")
        self.place_name_entry = ttk.Entry(place_frame, textvariable=self.place_name_var, width=40)
        self.place_name_entry.grid(row=0, column=1, padx=5, sticky="ew")
        ttk.Button(place_frame, text="Load Map & Train", command=self.run_real_world_training).grid(row=0, column=2, padx=5)
        place_frame.grid_columnconfigure(1, weight=1)

        display_buttons_frame_rw = ttk.Frame(control_frame_rw)
        display_buttons_frame_rw.pack(side="right", padx=5)
        ttk.Button(display_buttons_frame_rw, text="Show OSMnx Graph", command=self.show_osmnx_graph).pack(side="top", pady=2)
        ttk.Button(
            display_buttons_frame_rw, text="Show Q-Table", command=lambda: self.show_q_table_popup("real_world")
        ).pack(side="top", pady=2)

        self.progress_label_rw = ttk.Label(self.real_world_frame, text="Training Progress: 0%")
        self.progress_label_rw.pack(pady=2)
        self.progress_rw = ttk.Progressbar(self.real_world_frame, orient="horizontal", length=400, mode="determinate")
        self.progress_rw.pack(pady=5, fill="x", padx=5)

        self.fig_rw, (self.ax_rw_map, self.ax_rw_curve) = plt.subplots(
            2, 1, figsize=(30, 25), gridspec_kw={"height_ratios": [3, 1]}, dpi=100
        )
        self.canvas_rw_mpl = FigureCanvasTkAgg(self.fig_rw, master=self.real_world_frame)
        self.canvas_rw_mpl_widget = self.canvas_rw_mpl.get_tk_widget()
        self.canvas_rw_mpl_widget.pack(expand=True, fill="both", pady=10)

        self.ax_rw_map.set_title("Real-World Map Visualization")
        self.ax_rw_map.set_xticks([])
        self.ax_rw_map.set_yticks([])
        self.plot_learning_curve([], self.ax_rw_curve, self.canvas_rw_mpl, "Real-World Grid")

    def show_osmnx_graph(self):
        if not self.current_osm_graph:
            messagebox.showwarning("No Map Loaded", "Please load a real-world map first by clicking 'Load Map & Train'.")
            return

        import osmnx as ox

        fig, ax = ox.plot_graph(
            self.current_osm_graph, show=False, close=False, bgcolor="lightgray",
            edge_color="white", node_size=10, node_color="blue", edge_linewidth=0.5, figsize=(20, 20),
        )

        if self.real_world_graph_start_node and self.real_world_graph_end_node:
            s_node = self.real_world_graph_start_node
            e_node = self.real_world_graph_end_node

            if s_node in self.current_osm_graph.nodes:
                sx, sy = self.current_osm_graph.nodes[s_node]["x"], self.current_osm_graph.nodes[s_node]["y"]
                ax.scatter(sx, sy, color="green", s=200, zorder=5, edgecolors="black", label="Start Node")
                ax.text(sx, sy, "S", fontsize=12, ha="center", va="center", color="white", zorder=6)

            if e_node in self.current_osm_graph.nodes:
                ex, ey = self.current_osm_graph.nodes[e_node]["x"], self.current_osm_graph.nodes[e_node]["y"]
                ax.scatter(ex, ey, color="red", s=200, zorder=5, edgecolors="black", label="End Node")
                ax.text(ex, ey, "E", fontsize=12, ha="center", va="center", color="white", zorder=6)

        ax.set_title(f"OSMnx Graph for {self.place_name_var.get()}")
        plt.show()

    def run_real_world_training(self):
        self.stop_running_animation()

        self.clear_real_world_map_plot()
        self.progress_rw["value"] = 0
        self.progress_label_rw["text"] = "Training Progress: 0%"

        place_name = self.place_name_var.get().strip()
        if not place_name:
            messagebox.showwarning("Input Error", "Please enter a place name or 'Random Location'.")
            return

        try:
            learning_rate, discount_factor, epsilon_decay_rate, min_epsilon, total_episodes = (
                self.read_hyperparameters()
            )
        except ValueError as exc:
            messagebox.showerror("Input Error", f"Please enter valid numeric values for hyperparameters. Error: {exc}")
            return

        def real_world_thread():
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Loading Map & Training",
                    f"Starting to load map for '{place_name}' and train Q-Learning. "
                    "This may take some time depending on map size and internet speed.",
                ),
            )
            self.root.after(0, lambda: self.ax_rw_map.text(0.5, 0.5, "Loading Map...", ha="center", va="center", transform=self.ax_rw_map.transAxes))
            self.root.after(0, self.canvas_rw_mpl.draw)

            graph = get_real_world_graph(place_name=place_name)
            if not graph:
                self.root.after(0, lambda: messagebox.showerror(
                    "Graph Download Error",
                    f"Could not download graph for '{place_name}'. Common issues: incorrect place name, "
                    "no internet, or OSMnx not installed. See console for details.",
                ))
                self.root.after(0, lambda: self.update_progress(0, 1, "real_world"))
                self.root.after(0, self.clear_real_world_map_plot)
                return

            start_node_graph, end_node_graph = get_start_end_nodes(graph)
            if not start_node_graph or not end_node_graph:
                self.root.after(0, lambda: messagebox.showerror("Error", "Could not select valid start/end nodes from the map. Map might be too small or disconnected."))
                self.root.after(0, lambda: self.update_progress(0, 1, "real_world"))
                self.root.after(0, self.clear_real_world_map_plot)
                return

            self.current_osm_graph = graph
            self.real_world_graph_start_node = start_node_graph
            self.real_world_graph_end_node = end_node_graph

            (
                grid_size_real, start_state_real, end_state_real, obstacles_real,
                grid_to_node_map, node_to_grid_map, min_coords, max_coords, _, _,
            ) = graph_to_simplified_grid(graph, start_node_graph, end_node_graph, grid_size=REAL_WORLD_GRID_SIZE)

            if grid_size_real is None:
                self.root.after(0, lambda: messagebox.showerror("Error", "Could not create a valid grid from the map. Check console for details."))
                self.root.after(0, lambda: self.update_progress(0, 1, "real_world"))
                self.root.after(0, self.clear_real_world_map_plot)
                return

            self.current_grid_params = (grid_size_real, start_state_real, end_state_real, obstacles_real)
            self.real_world_grid_to_node_map = grid_to_node_map
            self.real_world_node_to_grid_map = node_to_grid_map
            self.real_world_min_coords = min_coords
            self.real_world_max_coords = max_coords

            q_table_real, episode_rewards_real = train_q_learning(
                grid_size=grid_size_real,
                start_state=start_state_real,
                end_state=end_state_real,
                obstacles=obstacles_real,
                total_episodes=total_episodes,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon_decay_rate=epsilon_decay_rate,
                min_epsilon=min_epsilon,
                progress_callback=lambda current, total: self.update_progress(current, total, "real_world"),
            )

            if q_table_real is not None:
                self.current_q_table = q_table_real
                self.current_tab_trained = "real_world"
                self.root.after(0, lambda: self.plot_learning_curve(episode_rewards_real, self.ax_rw_curve, self.canvas_rw_mpl, f"Real-World Grid ({place_name})"))
                self.root.after(0, lambda: messagebox.showinfo("Training Complete", f"Q-Learning training finished for Real-World Grid ('{place_name}'). Visualizing path..."))
                self.root.after(0, self.animate_real_world_path)
            else:
                self.root.after(0, lambda: messagebox.showerror("Training Failed", f"Q-Learning training for Real-World Grid ('{place_name}') failed or no path found. Check console for details."))
                self.root.after(0, lambda: self.plot_learning_curve([], self.ax_rw_curve, self.canvas_rw_mpl, f"Real-World Grid ({place_name})"))
                self.root.after(0, self.clear_real_world_map_plot)

        self.animation_thread = threading.Thread(target=real_world_thread, daemon=True)
        self.animation_thread.start()

    def clear_real_world_map_plot(self):
        self.ax_rw_map.clear()
        self.ax_rw_map.set_title("Real-World Map Visualization")
        self.ax_rw_map.set_xticks([])
        self.ax_rw_map.set_yticks([])
        self.ax_rw_curve.clear()
        self.ax_rw_curve.set_title("Real-World Grid Learning Curve")
        self.ax_rw_curve.set_xlabel("Episode")
        self.ax_rw_curve.set_ylabel("Total Reward")
        self.ax_rw_curve.grid(True)
        self.canvas_rw_mpl.draw()

    def draw_real_world_map(self, current_agent_pos_grid=None, optimal_path_grid=None):
        if not self.current_osm_graph or not self.current_grid_params:
            return

        import osmnx as ox

        grid_size, start_state_grid, end_state_grid, obstacles_grid = self.current_grid_params
        graph = self.current_osm_graph
        min_x, min_y = self.real_world_min_coords
        max_x, max_y = self.real_world_max_coords

        self.ax_rw_map.clear()
        self.ax_rw_map.set_title("Real-World Map Pathfinding")
        self.ax_rw_map.set_xticks([])
        self.ax_rw_map.set_yticks([])

        ox.plot_graph(
            graph, ax=self.ax_rw_map, show=False, close=False,
            bgcolor="white", edge_color="lightgray", node_size=0, edge_linewidth=0.5,
        )

        cell_width_map = (max_x - min_x) / grid_size
        cell_height_map = (max_y - min_y) / grid_size

        def grid_to_map_center_coords(r, c):
            map_x = min_x + c * cell_width_map + cell_width_map / 2
            map_y = max_y - r * cell_height_map - cell_height_map / 2
            return map_x, map_y

        for r, c in obstacles_grid:
            map_x_rect_origin = min_x + c * cell_width_map
            map_y_rect_origin = max_y - (r + 1) * cell_height_map
            self.ax_rw_map.add_patch(
                plt.Rectangle((map_x_rect_origin, map_y_rect_origin), cell_width_map, cell_height_map, color="darkgrey", alpha=0.5)
            )

        if self.real_world_graph_start_node and self.real_world_graph_start_node in graph.nodes:
            s_node = self.real_world_graph_start_node
            s_x, s_y = graph.nodes[s_node]["x"], graph.nodes[s_node]["y"]
            self.ax_rw_map.scatter(s_x, s_y, color="green", s=150, zorder=5, label="Start", edgecolors="black")
            self.ax_rw_map.text(s_x, s_y, "S", fontsize=10, ha="center", va="center", color="white", zorder=6)

        if self.real_world_graph_end_node and self.real_world_graph_end_node in graph.nodes:
            e_node = self.real_world_graph_end_node
            e_x, e_y = graph.nodes[e_node]["x"], graph.nodes[e_node]["y"]
            self.ax_rw_map.scatter(e_x, e_y, color="red", s=150, zorder=5, label="End", edgecolors="black")
            self.ax_rw_map.text(e_x, e_y, "E", fontsize=10, ha="center", va="center", color="white", zorder=6)

        if optimal_path_grid:
            real_path_coords = [grid_to_map_center_coords(r, c) for r, c in optimal_path_grid]
            if len(real_path_coords) > 1:
                xs = [p[0] for p in real_path_coords]
                ys = [p[1] for p in real_path_coords]
                self.ax_rw_map.plot(xs, ys, color="blue", linewidth=3, solid_capstyle="round", zorder=4, label="Optimal Path")
            elif len(real_path_coords) == 1:
                x, y = real_path_coords[0]
                self.ax_rw_map.scatter(x, y, color="blue", s=50, zorder=4, alpha=0.7)

        if current_agent_pos_grid:
            agent_x, agent_y = grid_to_map_center_coords(current_agent_pos_grid[0], current_agent_pos_grid[1])
            self.ax_rw_map.scatter(agent_x, agent_y, color="gold", s=200, zorder=10, marker="o", edgecolors="black")

        self.fig_rw.tight_layout(rect=[0, 0.1, 1, 1])
        self.canvas_rw_mpl.draw()

    def animate_real_world_path(self):
        if self.current_q_table is None or self.current_grid_params is None or self.current_tab_trained != "real_world":
            messagebox.showwarning("No Data", "Please load a map and train the Q-Learning model for the Real-World Map first.")
            return

        grid_size, start_state_grid, end_state_grid, obstacles_grid = self.current_grid_params
        optimal_path_grid = find_optimal_path(self.current_q_table, grid_size, start_state_grid, end_state_grid, obstacles_grid)

        if not optimal_path_grid:
            messagebox.showwarning("Path Not Found", "Optimal path could not be determined for the real-world map. Agent might be stuck or no path exists.")
            return

        self.root.after(0, lambda: self.draw_real_world_map(current_agent_pos_grid=None, optimal_path_grid=None))

        def animate():
            for state_grid in optimal_path_grid:
                if self.animation_stop_event.is_set():
                    print("Animation stopped by user.")
                    break
                self.root.after(0, lambda s_grid=state_grid: self.draw_real_world_map(current_agent_pos_grid=s_grid, optimal_path_grid=None))
                time.sleep(0.3)

            if not self.animation_stop_event.is_set():
                self.root.after(0, lambda: self.draw_real_world_map(current_agent_pos_grid=None, optimal_path_grid=optimal_path_grid))
                messagebox.showinfo("Animation Complete", "Agent has finished traversing the optimal path on the real-world map.")

        self.animation_thread = threading.Thread(target=animate, daemon=True)
        self.animation_thread.start()
