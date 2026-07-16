"""Dyna-Q: Q-Learning augmented with planning from a learned model.

Dyna-Q keeps everything plain Q-Learning does (learning directly from real
interaction with the environment) and adds a *model* of the environment. The
model simply remembers, for each ``(state, action)`` pair the agent has tried,
what reward and next state resulted::

    model[(state, action)] = (reward, next_state)

After every *real* step, the agent performs ``planning_steps`` extra
"imagined" updates: it samples previously-seen ``(state, action)`` pairs from
the model and applies the same Bellman update using the remembered outcome.
This lets the agent squeeze far more learning out of each real interaction, so
it typically converges in fewer real episodes than Q-Learning.

The environment here is deterministic, so a single remembered outcome per
``(state, action)`` pair is exact.
"""

import random

import numpy as np

from .config import DEFAULT_PLANNING_STEPS, NUM_ACTIONS
from .environment import get_next_state, get_reward, has_path


def _select_action(q_table, row, col, epsilon):
    """Epsilon-greedy action selection (ties broken uniformly at random)."""
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    q_values = q_table[row, col]
    max_q = np.max(q_values)
    best_actions = np.where(q_values == max_q)[0]
    return random.choice(best_actions)


def _bellman_update(q_table, state, action_index, reward, next_state, learning_rate, discount_factor):
    """Apply the standard Q-Learning (Bellman) update in place."""
    row, col = state
    next_row, next_col = next_state
    old_value = q_table[row, col, action_index]
    next_max_value = np.max(q_table[next_row, next_col])
    q_table[row, col, action_index] = old_value + learning_rate * (
        reward + discount_factor * next_max_value - old_value
    )


def train_dyna_q(
    grid_size,
    start_state,
    end_state,
    obstacles,
    total_episodes,
    learning_rate,
    discount_factor,
    epsilon_decay_rate,
    min_epsilon,
    planning_steps=DEFAULT_PLANNING_STEPS,
    change_episode=None,
    obstacles_after=None,
    progress_callback=None,
):
    """Run Dyna-Q and return ``(q_table, episode_rewards)``.

    Mirrors :func:`rl.q_learning.train_q_learning` but performs
    ``planning_steps`` model-based updates after each real step. Supports a
    dynamic environment via ``change_episode`` / ``obstacles_after`` (the
    learned model is intentionally *not* reset when the world changes, which
    is what makes the adaptation interesting). On failure (invalid inputs,
    start/end on an obstacle, or no path) it returns ``(None, [])``.
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
    epsilon = 1.0
    episode_rewards = []

    # The learned model and the list of observed (state, action) pairs we are
    # allowed to sample from during planning.
    model = {}
    observed_pairs = []

    print(f"Starting Dyna-Q training on {grid_size}x{grid_size} grid (planning_steps={planning_steps})...")

    for episode in range(total_episodes):
        # Dynamic environment: swap in the new obstacle layout. The stale
        # model entries are kept on purpose so we can observe how the agent
        # copes with an out-of-date world model.
        if change_episode is not None and episode == change_episode:
            obstacles = obstacles_after

        current_state = start_state
        done = False
        total_episode_reward = 0
        step_count = 0
        max_steps_per_episode = grid_size * grid_size * 4

        while not done and step_count < max_steps_per_episode:
            row, col = current_state

            # --- Act in the real environment -------------------------------
            action_index = _select_action(q_table, row, col, epsilon)
            next_state = get_next_state(current_state, action_index, grid_size, obstacles)
            reward = get_reward(next_state, end_state, obstacles)
            total_episode_reward += reward

            # Direct RL update from the real experience.
            _bellman_update(q_table, current_state, action_index, reward, next_state, learning_rate, discount_factor)

            # --- Update the model ------------------------------------------
            key = (current_state, action_index)
            if key not in model:
                observed_pairs.append(key)
            model[key] = (reward, next_state)

            # --- Planning: learn from imagined experience ------------------
            for _ in range(planning_steps):
                if not observed_pairs:
                    break
                (sim_state, sim_action) = random.choice(observed_pairs)
                sim_reward, sim_next_state = model[(sim_state, sim_action)]
                _bellman_update(
                    q_table, sim_state, sim_action, sim_reward, sim_next_state, learning_rate, discount_factor
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
