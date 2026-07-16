"""Reinforcement-learning core: environment, configuration and Q-Learning.

Nothing in this package imports GUI or mapping libraries, so it can be used
on its own from a script or a test.
"""

from .config import ACTIONS, NUM_ACTIONS
from .environment import (
    generate_obstacles,
    get_next_state,
    get_reward,
    has_path,
    is_valid_state,
    make_extra_obstacles,
)
from .q_learning import find_optimal_path, train_q_learning
from .dyna_q import train_dyna_q
from .dyna_q_plus import train_dyna_q_plus
from .prioritized_sweeping import train_prioritized_sweeping
from .algorithms import ALGORITHMS, ALGORITHM_COLORS

__all__ = [
    "ACTIONS",
    "NUM_ACTIONS",
    "is_valid_state",
    "get_next_state",
    "get_reward",
    "has_path",
    "generate_obstacles",
    "make_extra_obstacles",
    "train_q_learning",
    "train_dyna_q",
    "train_dyna_q_plus",
    "train_prioritized_sweeping",
    "find_optimal_path",
    "ALGORITHMS",
    "ALGORITHM_COLORS",
]
