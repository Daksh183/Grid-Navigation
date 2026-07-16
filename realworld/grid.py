"""Convert a real-world street graph into a simplified grid for Q-Learning.

The Q-Learning core works on a square grid, so a street network's node
coordinates are normalised and binned into ``grid_size`` x ``grid_size``
cells. Cells that contain no street node become obstacles.

All failure paths return a tuple of ten ``None`` values so that callers can
safely unpack the result before checking whether it succeeded.
"""

import numpy as np

# Consistent "failed" return so the 10-value unpack at the call site never
# raises before the caller gets a chance to check for None.
_FAILURE = (None,) * 10


def graph_to_simplified_grid(graph, start_node_graph, end_node_graph, grid_size=30):
    """Bin a street graph into a grid usable by the Q-Learning core.

    Returns a 10-tuple::

        (grid_size, start_grid_state, end_grid_state, obstacles,
         grid_to_node_map, node_to_grid_map, min_coords, max_coords,
         start_node_graph, end_node_graph)

    or ten ``None`` values on failure.
    """
    if not graph or start_node_graph is None or end_node_graph is None:
        print("Graph or start/end nodes invalid for grid conversion.")
        return _FAILURE

    nodes_data = {node: data for node, data in graph.nodes(data=True)}

    if start_node_graph not in nodes_data or end_node_graph not in nodes_data:
        print("Start or end node not found in graph data.")
        return _FAILURE

    coords = np.array([(data["x"], data["y"]) for data in nodes_data.values()])
    if len(coords) == 0:
        print("No node coordinates found in the graph.")
        return _FAILURE

    min_x, min_y = np.min(coords, axis=0)
    max_x, max_y = np.max(coords, axis=0)

    # Guard against a graph with no spatial extent (division by zero).
    range_x = max(max_x - min_x, 1e-6)
    range_y = max(max_y - min_y, 1e-6)

    node_to_grid_map = {}
    grid_to_node_map = {}

    for node, data in nodes_data.items():
        grid_state = _map_coord_to_grid(
            data["x"], data["y"], min_x, max_y, grid_size, range_x, range_y
        )
        node_to_grid_map[node] = grid_state
        # Keep the first node mapped to each cell for path reconstruction.
        grid_to_node_map.setdefault(grid_state, node)

    # Any cell without a mapped node is treated as an obstacle.
    all_grid_cells = {(r, c) for r in range(grid_size) for c in range(grid_size)}
    mapped_grid_cells = set(grid_to_node_map.keys())
    grid_obstacles = all_grid_cells - mapped_grid_cells

    start_grid_state = node_to_grid_map.get(start_node_graph)
    end_grid_state = node_to_grid_map.get(end_node_graph)

    # Rounding can drop the start/end onto an obstacle cell; snap to nearest.
    if start_grid_state in grid_obstacles:
        start_grid_state = find_nearest_mapped_cell(start_grid_state, mapped_grid_cells)
        if start_grid_state is None:
            print("Could not find a valid start grid state.")
            return _FAILURE
        start_node_graph = grid_to_node_map[start_grid_state]

    if end_grid_state in grid_obstacles:
        end_grid_state = find_nearest_mapped_cell(end_grid_state, mapped_grid_cells)
        if end_grid_state is None:
            print("Could not find a valid end grid state.")
            return _FAILURE
        end_node_graph = grid_to_node_map[end_grid_state]

    if start_grid_state == end_grid_state:
        print(f"Start and end map to the same grid cell: {start_grid_state}.")
        return _FAILURE

    grid_obstacles.discard(start_grid_state)
    grid_obstacles.discard(end_grid_state)

    print(f"Simplified grid size: {grid_size}x{grid_size}")
    print(f"Start grid state: {start_grid_state}")
    print(f"End grid state: {end_grid_state}")
    print(f"Number of obstacles in grid: {len(grid_obstacles)}")

    return (
        grid_size,
        start_grid_state,
        end_grid_state,
        list(grid_obstacles),
        grid_to_node_map,
        node_to_grid_map,
        (min_x, min_y),
        (max_x, max_y),
        start_node_graph,
        end_node_graph,
    )


def find_nearest_mapped_cell(target_cell, mapped_cells):
    """Return the cell in ``mapped_cells`` closest (Euclidean) to ``target_cell``."""
    if not mapped_cells:
        return None

    target_row, target_col = target_cell
    nearest_cell = None
    min_dist_sq = float("inf")

    for row, col in mapped_cells:
        dist_sq = (row - target_row) ** 2 + (col - target_col) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            nearest_cell = (row, col)
    return nearest_cell


def _map_coord_to_grid(x, y, min_x, max_y, grid_dim, range_x, range_y):
    """Map a real-world (x, y) coordinate to a (row, col) grid cell.

    The y-axis is inverted so that higher latitudes appear at the top of the
    grid, matching the on-screen visualisation.
    """
    row = int(((max_y - y) / range_y) * grid_dim)
    col = int(((x - min_x) / range_x) * grid_dim)
    row = max(0, min(grid_dim - 1, row))
    col = max(0, min(grid_dim - 1, col))
    return row, col
