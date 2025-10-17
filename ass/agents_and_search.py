
# agents_and_search.py
import heapq
from collections import deque

# We expect the graph to be the junction graph returned by get_junction_graph
# Nodes are tuples (x,y). Edges include path cost.

def uniform_cost_search(graph, start, goal):
    """
    graph: {node: [(neighbor, path_cells, path_actions, cost), ...]}
    returns path as list of nodes (junction nodes) and the low-level path (sequence of cells)
    """
    frontier = []
    heapq.heappush(frontier, (0, start, [start], []))  # (path_cost, node, node_path, low_level_cells)
    explored = {}
    while frontier:
        cost, node, node_path, low_cells = heapq.heappop(frontier)
        if node == goal:
            return node_path, low_cells
        if node in explored and explored[node] <= cost:
            continue
        explored[node] = cost
        for neigh, cells, actions, ecost in graph.get(node, []):
            new_cost = cost + ecost
            heapq.heappush(frontier, (new_cost, neigh, node_path + [neigh], low_cells + cells))
    return None, None

def depth_limited_search_recursive(graph, node, goal, limit, visited_nodes):
    """
    Recursive DLS on junction graph measured by number of junction-nodes depth.
    Returns:
      'cutoff' (if cutoff occurred but no solution),
      None for failure,
      solution tuple (node_path, low_level_cells)
    """
    if node == goal:
        return [node], []  # reached goal at node-level; low-level path empty (handled by caller)
    if limit == 0:
        return 'cutoff'
    cutoff_occurred = False
    for neigh, cells, actions, cost in graph.get(node, []):
        if neigh in visited_nodes:
            continue
        visited_nodes.add(neigh)
        result = depth_limited_search_recursive(graph, neigh, goal, limit - 1, visited_nodes)
        visited_nodes.remove(neigh)
        if result == 'cutoff':
            cutoff_occurred = True
        elif result is not None:  # success
            node_path, low = result
            return [node] + node_path, cells + low  # prepend current node and corridor cells
    return 'cutoff' if cutoff_occurred else None

def iterative_deepening_search(graph, start, goal, max_depth=50):
    for depth in range(0, max_depth + 1):
        visited = set([start])
        result = depth_limited_search_recursive(graph, start, goal, depth, visited)
        if result == 'cutoff':
            continue
        if result is not None:
            # result returns node_path and low-level cells aggregated; might be incomplete low-level, but ok
            return result
    return None
