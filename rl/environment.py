"""Grid environment logic for Q-Learning pathfinding.

This module is intentionally free of any GUI code so that the environment
can be reused, tested, or driven from a headless script.
"""

import collections
import random

from .config import (
    ACTIONS,
    DEFAULT_REWARD,
    GOAL_REWARD,
    OBSTACLE_REWARD,
)


def is_valid_state(state, grid_size, obstacles):
    """Return True if ``state`` is inside the grid and not an obstacle."""
    row, col = state
    return 0 <= row < grid_size and 0 <= col < grid_size and state not in obstacles


def get_next_state(current_state, action_index, grid_size, obstacles):
    """Apply an action and return the resulting state.

    If the move would leave the grid or hit an obstacle, the agent stays
    in its current cell.
    """
    row, col = current_state
    dr, dc = ACTIONS[action_index]
    next_state = (row + dr, col + dc)

    if is_valid_state(next_state, grid_size, obstacles):
        return next_state
    return current_state


def get_reward(state, end_state, obstacles):
    """Return the reward for arriving at ``state``."""
    if state == end_state:
        return GOAL_REWARD
    if state in obstacles:
        return OBSTACLE_REWARD
    return DEFAULT_REWARD


def has_path(grid_size, start, end, obstacles):
    """Check whether a path exists from ``start`` to ``end`` using BFS."""
    queue = collections.deque([start])
    visited = {start}

    while queue:
        r, c = queue.popleft()
        if (r, c) == end:
            return True

        for dr, dc in ACTIONS:
            next_state = (r + dr, c + dc)
            nr, nc = next_state
            if (
                0 <= nr < grid_size
                and 0 <= nc < grid_size
                and next_state not in obstacles
                and next_state not in visited
            ):
                visited.add(next_state)
                queue.append(next_state)
    return False


def generate_obstacles(grid_size, num_obstacles, start_state, end_state):
    """Randomly place obstacles while guaranteeing a start-to-end path.

    Returns the list of placed obstacle cells. The number placed may be
    fewer than requested if the grid becomes too constrained; callers can
    compare ``len(result)`` against ``num_obstacles`` to detect this.
    """
    all_cells = [(r, c) for r in range(grid_size) for c in range(grid_size)]
    available_cells = [
        cell for cell in all_cells if cell != start_state and cell != end_state
    ]

    obstacles = set()
    tried_cells = set()

    # Give more attempts than requested so that rejected (path-blocking)
    # placements do not prevent us from reaching the target count.
    for _ in range(num_obstacles * 5):
        if len(obstacles) >= num_obstacles:
            break
        if not available_cells:
            break

        chosen = random.choice(available_cells)
        if chosen in tried_cells:
            continue
        tried_cells.add(chosen)

        # Only keep the obstacle if a path still exists afterwards.
        if has_path(grid_size, start_state, end_state, list(obstacles | {chosen})):
            obstacles.add(chosen)
        available_cells.remove(chosen)

    return list(obstacles)


def make_extra_obstacles(grid_size, base_obstacles, start_state, end_state, num_extra):
    """Return up to ``num_extra`` new obstacle cells to add to ``base_obstacles``.

    Every returned cell is chosen so that ``base_obstacles`` plus all the
    returned cells still leaves a start-to-end path. This is used to build the
    "before"/"after" layouts for a dynamic environment (adding these cells
    blocks routes; removing them opens a shortcut).
    """
    base = set(base_obstacles)
    candidates = [
        (r, c)
        for r in range(grid_size)
        for c in range(grid_size)
        if (r, c) not in base and (r, c) != start_state and (r, c) != end_state
    ]
    random.shuffle(candidates)

    extra = set()
    for cell in candidates:
        if len(extra) >= num_extra:
            break
        if has_path(grid_size, start_state, end_state, list(base | extra | {cell})):
            extra.add(cell)
    return list(extra)
