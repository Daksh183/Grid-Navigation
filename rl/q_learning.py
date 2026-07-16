"""Tabular Q-Learning training and optimal-path extraction.

This is the heart of the project: an agent learns a Q-table by interacting
with the grid :mod:`environment`, then the greedy policy is followed to
recover the optimal path.

The functions here are GUI-free. Progress is reported through an optional
``progress_callback(current_episode, total_episodes)`` so a front-end can
display a progress bar without this module knowing anything about it.
"""

import random

import numpy as np

from .config import NUM_ACTIONS
from .environment import get_next_state, get_reward, has_path


def train_q_learning(
    grid_size,
    start_state,
    end_state,
    obstacles,
    total_episodes,
    learning_rate,
    discount_factor,
    epsilon_decay_rate,
    min_epsilon,
    change_episode=None,
    obstacles_after=None,
    progress_callback=None,
):
    """Run Q-Learning and return ``(q_table, episode_rewards)``.

    For a dynamic environment, pass ``change_episode`` and ``obstacles_after``:
    at the start of that episode the obstacle layout switches to
    ``obstacles_after`` (both layouts must be solvable). Leave them as ``None``
    for a static environment.

    On failure (invalid inputs, start/end on an obstacle, or no path) the
    function returns ``(None, [])`` after printing the reason.
    """
    if not grid_size or start_state is None or end_state is None or obstacles is None:
        print("Cannot train: Invalid grid parameters.")
        return None, []

    if start_state in obstacles or end_state in obstacles:
        print("Start or End state is an obstacle. Training aborted.")
        return None, []

    if not has_path(grid_size, start_state, end_state, obstacles):
        print("No path exists between start and end with given obstacles. Training aborted.")
        return None, []

    q_table = np.zeros((grid_size, grid_size, NUM_ACTIONS))
    epsilon = 1.0  # start fully exploratory
    episode_rewards = []

    print(f"Starting Q-Learning training on {grid_size}x{grid_size} grid...")

    for episode in range(total_episodes):
        # Dynamic environment: swap in the new obstacle layout at the
        # configured episode. The agent must adapt to the change.
        if change_episode is not None and episode == change_episode:
            obstacles = obstacles_after

        current_state = start_state
        done = False
        total_episode_reward = 0
        step_count = 0
        # Cap steps so difficult grids cannot loop forever.
        max_steps_per_episode = grid_size * grid_size * 4

        while not done and step_count < max_steps_per_episode:
            row, col = current_state

            # Epsilon-greedy action selection.
            if random.uniform(0, 1) < epsilon:
                action_index = random.randint(0, NUM_ACTIONS - 1)
            else:
                q_values = q_table[row, col]
                max_q = np.max(q_values)
                best_actions = np.where(q_values == max_q)[0]
                action_index = random.choice(best_actions)

            next_state = get_next_state(current_state, action_index, grid_size, obstacles)
            reward = get_reward(next_state, end_state, obstacles)
            total_episode_reward += reward

            # Bellman update.
            next_row, next_col = next_state
            old_value = q_table[row, col, action_index]
            next_max_value = np.max(q_table[next_row, next_col])
            q_table[row, col, action_index] = old_value + learning_rate * (
                reward + discount_factor * next_max_value - old_value
            )

            current_state = next_state
            if current_state == end_state:
                done = True
            step_count += 1

        epsilon = max(min_epsilon, epsilon * epsilon_decay_rate)
        episode_rewards.append(total_episode_reward)

        if progress_callback:
            progress_callback(episode + 1, total_episodes)

    print("Training finished.")
    return q_table, episode_rewards


def find_optimal_path(q_table, grid_size, start_state, end_state, obstacles):
    """Greedily follow the trained Q-table to reconstruct the optimal path."""
    if (
        q_table is None
        or grid_size is None
        or start_state is None
        or end_state is None
        or obstacles is None
    ):
        print("Cannot find path: Invalid input.")
        return []

    print("\nFinding optimal path using the trained Q-table:")
    current_state = start_state
    optimal_path = [current_state]
    step_count = 0
    max_path_length = grid_size * grid_size * 2

    while current_state != end_state and step_count < max_path_length:
        row, col = current_state

        if not (0 <= row < grid_size and 0 <= col < grid_size):
            print(f"Agent went out of bounds at {current_state}. Pathfinding stopped.")
            break

        if (row, col) in obstacles:
            print(f"Agent stepped into an obstacle at {current_state}. Pathfinding stopped.")
            break

        q_values_for_state = q_table[row, col]
        if np.all(q_values_for_state == q_values_for_state[0]):
            # No preference learned for this state yet: pick any valid move.
            valid_actions = _valid_action_indices(row, col, grid_size, obstacles)
            if valid_actions:
                action_index = random.choice(valid_actions)
            else:
                print(f"Agent is trapped at {current_state}. Cannot reach goal.")
                break
        else:
            action_index = np.argmax(q_values_for_state)

        next_state = get_next_state(current_state, action_index, grid_size, obstacles)

        # Unable to move and not at the goal: the agent is stuck.
        if next_state == current_state and current_state != end_state:
            print(f"Agent stuck at {current_state}. Cannot reach goal.")
            break

        current_state = next_state
        optimal_path.append(current_state)
        step_count += 1

    if current_state == end_state:
        print("Goal reached.")
    else:
        print("Pathfinding terminated without reaching goal (might be stuck or path too long).")

    return optimal_path


def _valid_action_indices(row, col, grid_size, obstacles):
    """Return the indices of actions that lead to a valid, unblocked cell."""
    from .config import ACTIONS  # local import to keep the public surface tidy

    valid = []
    for i, (dr, dc) in enumerate(ACTIONS):
        nr, nc = row + dr, col + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size and (nr, nc) not in obstacles:
            valid.append(i)
    return valid
