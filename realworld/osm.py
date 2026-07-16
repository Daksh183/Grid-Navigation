"""Download and inspect real-world street networks via OSMnx.

``osmnx`` is imported lazily inside the functions so that the rest of the
application (in particular the grid-only mode) runs even when OSMnx and its
heavy geospatial dependencies are not installed.
"""

import random


def get_real_world_graph(place_name="Random Location"):
    """Download a drivable street-network graph for ``place_name``.

    A special value of ``"Random Location"`` loads a small, fixed area near
    the San Francisco Civic Center, which keeps the download small and the
    result reproducible. Returns ``None`` on failure.
    """
    import osmnx as ox

    try:
        print(f"Attempting to download graph for: {place_name}")
        if place_name.lower() == "random location":
            lat, lon = 37.7849, -122.4172  # near Civic Center, San Francisco
            graph = ox.graph_from_point((lat, lon), dist=500, network_type="drive")
            print(f"Downloaded graph for a random location near ({lat:.4f}, {lon:.4f})")
        else:
            graph = ox.graph_from_place(place_name, network_type="drive")
            print(f"Downloaded graph for {place_name}")
        return graph
    except Exception as exc:  # noqa: BLE001 - surface any OSMnx/network error
        print(f"Error downloading graph: {exc}")
        return None


def get_start_end_nodes(graph):
    """Pick two distinct random nodes to use as start and end points."""
    if not graph or not graph.nodes():
        print("Graph is empty or invalid.")
        return None, None

    nodes = list(graph.nodes())
    if len(nodes) < 2:
        print("Graph has too few nodes to select start/end.")
        return None, None

    start_node = random.choice(nodes)
    end_node = random.choice(nodes)
    while start_node == end_node:
        end_node = random.choice(nodes)

    print(f"Selected graph start node: {start_node}, End node: {end_node}")
    return start_node, end_node
