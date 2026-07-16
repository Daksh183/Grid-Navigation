"""Registry of the available training algorithms.

Having one ordered mapping of display-name -> trainer lets the GUI build its
algorithm checkboxes and comparison runs without hard-coding each algorithm.
``supports`` lists the algorithm-specific optional parameters each trainer
accepts (all trainers additionally accept ``change_episode``,
``obstacles_after`` and ``progress_callback``).
"""

from .dyna_q import train_dyna_q
from .dyna_q_plus import train_dyna_q_plus
from .prioritized_sweeping import train_prioritized_sweeping
from .q_learning import train_q_learning

ALGORITHMS = {
    "Q-Learning": {"fn": train_q_learning, "supports": set()},
    "Dyna-Q": {"fn": train_dyna_q, "supports": {"planning_steps"}},
    "Dyna-Q+": {"fn": train_dyna_q_plus, "supports": {"planning_steps", "kappa"}},
    "Prioritized Sweeping": {"fn": train_prioritized_sweeping, "supports": {"planning_steps", "theta"}},
}

# Distinct colours for the learning curves and grid labels, keyed by algorithm
# name. Hex codes (the Matplotlib "tab" palette) so they work for both
# Matplotlib plotting and Tkinter widgets.
ALGORITHM_COLORS = {
    "Q-Learning": "#1f77b4",
    "Dyna-Q": "#2ca02c",
    "Dyna-Q+": "#ff7f0e",
    "Prioritized Sweeping": "#d62728",
}
