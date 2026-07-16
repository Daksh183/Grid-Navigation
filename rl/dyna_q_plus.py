"""Dyna-Q+: Dyna-Q with an exploration bonus for stale state-action pairs.

Dyna-Q+ builds directly on Dyna-Q and adds two changes that push the agent to
keep exploring, which is especially valuable when the environment can change
over time:

1. **Track recency.** For every ``(state, action)`` pair it records ``tau`` --
   the number of real time steps since that pair was last actually tried.

2. **Exploration bonus.** During planning, the remembered reward is boosted by
   ``kappa * sqrt(tau)``. Pairs that have not been tried for a long time look
   more attractive, so the agent periodically revisits neglected regions.

To make even *never-tried* actions candidates for exploration, the first time
a state is visited every action from it is seeded into the model as a
"stay put, reward 0" transition. The bonus then makes those untried actions
worth investigating.

Note: the exploration bonus is used **only** for planning updates and for
steering behaviour; the returned per-episode reward counts real reward only,
so it stays directly comparable with Q-Learning and Dyna-Q.
"""

import math
import random

import numpy as np

from .config import DEFAULT_KAPPA, DEFAULT_PLANNING_STEPS, NUM_ACTIONS
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


def train_dyna_q_plus(
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
    kappa=DEFAULT_KAPPA,
    change_episode=None,
    obstacles_after=None,
    progress_callback=None,
):
    """Run Dyna-Q+ and return ``(q_table, episode_rewards)``.

    Same contract as :func:`rl.dyna_q.train_dyna_q`, with an additional
    ``kappa`` controlling the exploration-bonus strength and support for a
    dynamic environment via ``change_episode`` / ``obstacles_after``. This is
    the setting Dyna-Q+ is designed for: the exploration bonus drives the
    agent to rediscover routes after the world changes. Returns ``(None, [])``
    on invalid inputs / no path.
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

    model = {}               # (state, action) -> (reward, next_state)
    observed_pairs = []      # keys of `model`, for uniform sampling in planning
    last_tried = {}          # (state, action) -> global step it was last tried
    seen_states = set()      # states whose full action set has been seeded
    global_step = 0          # counts every real environment step

    print(f"Starting Dyna-Q+ training on {grid_size}x{grid_size} grid "
          f"(planning_steps={planning_steps}, kappa={kappa})...")

    for episode in range(total_episodes):
        # Dynamic environment: swap the obstacle layout. Stale model entries
        # are kept; the exploration bonus is what helps the agent recover.
        if change_episode is not None and episode == change_episode:
            obstacles = obstacles_after

        current_state = start_state
        done = False
        total_episode_reward = 0
        step_count = 0
        max_steps_per_episode = grid_size * grid_size * 4

        while not done and step_count < max_steps_per_episode:
            global_step += 1
            row, col = current_state

            # --- Act in the real environment -------------------------------
            action_index = _select_action(q_table, row, col, epsilon)
            next_state = get_next_state(current_state, action_index, grid_size, obstacles)
            reward = get_reward(next_state, end_state, obstacles)
            total_episode_reward += reward

            _bellman_update(q_table, current_state, action_index, reward, next_state, learning_rate, discount_factor)

            # --- Seed every action of a newly-seen state -------------------
            # Untried actions are modelled as "stay put, reward 0" so the
            # exploration bonus can later make them worth trying.
            if current_state not in seen_states:
                seen_states.add(current_state)
                for a in range(NUM_ACTIONS):
                    key = (current_state, a)
                    model[key] = (0.0, current_state)
                    observed_pairs.append(key)
                    last_tried[key] = 0

            # --- Record the real transition --------------------------------
            key = (current_state, action_index)
            model[key] = (reward, next_state)
            last_tried[key] = global_step

            # --- Planning with exploration bonus ---------------------------
            for _ in range(planning_steps):
                if not observed_pairs:
                    break
                sim_state, sim_action = random.choice(observed_pairs)
                sim_reward, sim_next_state = model[(sim_state, sim_action)]
                tau = global_step - last_tried[(sim_state, sim_action)]
                bonus = kappa * math.sqrt(tau)
                _bellman_update(
                    q_table, sim_state, sim_action, sim_reward + bonus, sim_next_state,
                    learning_rate, discount_factor,
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
