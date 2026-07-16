"""Real-world map integration built on OpenStreetMap data (via OSMnx).

OSMnx is imported lazily by the functions in :mod:`realworld.osm`, so simply
importing this package does not require OSMnx to be installed.
"""

from .grid import find_nearest_mapped_cell, graph_to_simplified_grid
from .osm import get_real_world_graph, get_start_end_nodes

__all__ = [
    "get_real_world_graph",
    "get_start_end_nodes",
    "graph_to_simplified_grid",
    "find_nearest_mapped_cell",
]
