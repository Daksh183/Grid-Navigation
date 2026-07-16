"""Metrics for comparing reinforcement-learning algorithms.

These summarise a training run so different algorithms can be ranked on the
criteria the project's report cares about: how fast they converge, how good
and how stable the final policy is, and how long training took.
"""

import numpy as np


def moving_average(values, window=10):
    """Return the simple moving average of ``values`` (for smoother plots)."""
    if not values:
        return []
    window = max(1, min(window, len(values)))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid").tolist()


def convergence_episode(episode_rewards, window=20, tolerance=5.0):
    """Estimate the episode at which learning effectively converged.

    Returns the first episode after which the ``window``-episode moving
    average stays within ``tolerance`` of the final average reward. Fewer is
    better. Returns the episode count if it never settles.
    """
    n = len(episode_rewards)
    if n <= window:
        return n

    rewards = np.asarray(episode_rewards, dtype=float)
    final_avg = np.mean(rewards[-window:])
    for i in range(n - window):
        if abs(np.mean(rewards[i:i + window]) - final_avg) <= tolerance:
            return i
    return n


def final_average_reward(episode_rewards, window=20):
    """Average reward over the last ``window`` episodes (higher is better)."""
    if not episode_rewards:
        return float("nan")
    window = min(window, len(episode_rewards))
    return float(np.mean(episode_rewards[-window:]))


def stability(episode_rewards, window=20):
    """Std-dev of reward over the last ``window`` episodes (lower = steadier)."""
    if not episode_rewards:
        return float("nan")
    window = min(window, len(episode_rewards))
    return float(np.std(episode_rewards[-window:]))


def summarize(episode_rewards, path=None, end_state=None, train_time=None, window=20):
    """Bundle the key metrics for one algorithm run into a dict."""
    reached_goal = None
    path_length = None
    if path is not None:
        path_length = len(path)
        if end_state is not None:
            reached_goal = bool(path) and path[-1] == end_state

    return {
        "convergence_episode": convergence_episode(episode_rewards, window=window),
        "final_reward": final_average_reward(episode_rewards, window=window),
        "stability": stability(episode_rewards, window=window),
        "path_length": path_length,
        "reached_goal": reached_goal,
        "train_time": train_time,
    }
