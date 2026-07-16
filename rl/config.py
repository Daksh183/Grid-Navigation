"""Shared constants and default hyperparameters for the Q-Learning core.

Keeping these in one place makes it easy to tune the environment and the
learning algorithm without hunting through the rest of the code base.
"""

# --- Actions -----------------------------------------------------------------
# The agent can move in 8 directions (4 straight + 4 diagonal).
# Each action is expressed as a (row_delta, col_delta) pair.
ACTIONS = [
    (0, -1),   # left
    (0, 1),    # right
    (-1, 0),   # up
    (1, 0),    # down
    (-1, -1),  # up-left
    (-1, 1),   # up-right
    (1, -1),   # down-left
    (1, 1),    # down-right
]
NUM_ACTIONS = len(ACTIONS)

# --- Reward structure --------------------------------------------------------
GOAL_REWARD = 100      # reward for reaching the target tile
OBSTACLE_REWARD = -10  # penalty for landing on an obstacle
DEFAULT_REWARD = -1    # small step penalty to encourage short paths

# --- Default hyperparameters (used to seed the GUI fields) -------------------
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_DISCOUNT_FACTOR = 0.99
DEFAULT_EPSILON_DECAY_RATE = 0.9995
DEFAULT_MIN_EPSILON = 0.01
DEFAULT_TOTAL_EPISODES = 5000

# Number of simulated "planning" updates performed per real step by the
# model-based algorithms (Dyna-Q, Dyna-Q+, Prioritized Sweeping).
DEFAULT_PLANNING_STEPS = 10

# Dyna-Q+ exploration-bonus weight (kappa). During planning, a bonus of
# kappa * sqrt(time_since_last_tried) is added to the reward, encouraging the
# agent to revisit state-action pairs it has not tried in a while. Keep it
# small so it nudges exploration without swamping the real rewards.
DEFAULT_KAPPA = 1e-3

# Prioritized Sweeping priority threshold (theta). A state-action pair is only
# added to the priority queue when its temporal-difference error exceeds this
# value, so tiny, negligible updates are skipped.
DEFAULT_THETA = 1e-4

# Grid size used when a real-world map is discretised into a grid.
REAL_WORLD_GRID_SIZE = 30
