"""Prioritized Sweeping: focus planning effort where it matters most.

Dyna-Q replays *random* remembered transitions during planning. Most of those
updates change almost nothing. Prioritized Sweeping instead keeps a priority
queue of state-action pairs ordered by the size of their temporal-difference
(TD) error and always updates the most "surprising" pair first. Crucially,
after updating a pair it looks at that state's **predecessors** -- the pairs
known to lead into it -- and queues any whose value would now shift
meaningfully. This propagates information backward from the goal along the
useful paths, so the algorithm converges in far fewer updates.

Key data structures:

* ``model[(s, a)] = (reward, next_state)`` -- the learned environment model.
* ``predecessors[s] = {(s_prev, a_prev), ...}`` -- every pair seen to lead to ``s``.
* a max-priority queue keyed on absolute TD-error, with lazy invalidation so a
  pair's priority can be raised without removing its stale queue entry.
"""

import heapq
import itertools
import random

import numpy as np

from .config import DEFAULT_PLANNING_STEPS, DEFAULT_THETA, NUM_ACTIONS
from .environment import get_next_state, get_reward, has_path


def _select_action(q_table, row, col, epsilon):
    """Epsilon-greedy action selection (ties broken uniformly at random)."""
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    q_values = q_table[row, col]
    max_q = np.max(q_values)
    best_actions = np.where(q_values == max_q)[0]
    return random.choice(best_actions)


def _td_error(q_table, state, action_index, reward, next_state, discount_factor):
    """Absolute temporal-difference error for a transition (used as priority)."""
    row, col = state
    next_row, next_col = next_state
    best_next = np.max(q_table[next_row, next_col])
    return abs(reward + discount_factor * best_next - q_table[row, col, action_index])


class _PriorityQueue:
    """Max-priority queue with lazy invalidation for (state, action) pairs.

    ``heapq`` is a min-heap, so priorities are stored negated. A pair may
    appear multiple times; ``_current`` records each pair's live priority and
    stale heap entries are skipped on pop.
    """

    def __init__(self):
        self._heap = []
        self._current = {}
        self._counter = itertools.count()  # stable tie-breaker

    def push(self, sa, priority):
        # Keep only the highest priority requested for a given pair.
        if sa in self._current and self._current[sa] >= priority:
            return
        self._current[sa] = priority
        heapq.heappush(self._heap, (-priority, next(self._counter), sa))

    def pop(self):
        while self._heap:
            neg_priority, _, sa = heapq.heappop(self._heap)
            if self._current.get(sa) == -neg_priority:
                del self._current[sa]
                return sa
        return None

    def __bool__(self):
        return bool(self._current)


def train_prioritized_sweeping(
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
    theta=DEFAULT_THETA,
    change_episode=None,
    obstacles_after=None,
    progress_callback=None,
):
    """Run Prioritized Sweeping and return ``(q_table, episode_rewards)``.

    Same contract as the other trainers, with ``theta`` controlling the
    minimum TD-error required to enqueue a pair, plus dynamic-environment
    support via ``change_episode`` / ``obstacles_after``. Returns
    ``(None, [])`` on invalid inputs / no path.
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

    model = {}                    # (state, action) -> (reward, next_state)
    predecessors = {}             # state -> set of (prev_state, prev_action)
    pqueue = _PriorityQueue()

    print(f"Starting Prioritized Sweeping on {grid_size}x{grid_size} grid "
          f"(planning_steps={planning_steps}, theta={theta})...")

    for episode in range(total_episodes):
        # Dynamic environment: swap the obstacle layout. The model and
        # predecessor maps keep their (now partly stale) entries.
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

            # --- Update model and predecessor bookkeeping ------------------
            model[(current_state, action_index)] = (reward, next_state)
            predecessors.setdefault(next_state, set()).add((current_state, action_index))

            # --- Enqueue the real transition if it is surprising enough -----
            priority = _td_error(q_table, current_state, action_index, reward, next_state, discount_factor)
            if priority > theta:
                pqueue.push((current_state, action_index), priority)

            # --- Prioritized planning sweep --------------------------------
            for _ in range(planning_steps):
                sa = pqueue.pop()
                if sa is None:
                    break
                s, a = sa
                r, s_next = model[(s, a)]

                # Apply the (highest-priority) Bellman update.
                sr, sc = s
                nr, nc = s_next
                best_next = np.max(q_table[nr, nc])
                q_table[sr, sc, a] += learning_rate * (
                    r + discount_factor * best_next - q_table[sr, sc, a]
                )

                # Propagate to predecessors of s: they may now need updating.
                for prev_state, prev_action in predecessors.get(s, ()):  # empty if none known
                    prev_reward, _ = model[(prev_state, prev_action)]
                    prev_priority = _td_error(
                        q_table, prev_state, prev_action, prev_reward, s, discount_factor
                    )
                    if prev_priority > theta:
                        pqueue.push((prev_state, prev_action), prev_priority)

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
