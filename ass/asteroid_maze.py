
# asteroid_maze.py
import numpy as np
import random
from collections import deque, defaultdict

ACTION_MOVES = {0: (0, -1), 1: (-1, 0), 2: (0, 1), 3: (1, 0)}
# action costs for path-cost (left,right:2; down:1; up:4)
ACTION_COST = {0: 2, 2: 2, 3: 1, 1: 4}

def make_maze(n=7, asteroid_density=0.25, enemy_density=0.10, seed=None):
    """
    Create n x n maze:
      0 = asteroid (wall)
      1 = free
      3 = enemy (will be set later)
    Returns maze np.array, list of enemy coords (empty for now)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    else:
        np.random.seed(None)
        random.seed(None)

    maze = np.ones((n, n), dtype=int)
    # add asteroids
    for i in range(n):
        for j in range(n):
            if np.random.rand() < asteroid_density:
                maze[i, j] = 0

    # mark enemies after we ensure free cells exist
    # We'll select enemy positions later (separately)
    return maze

def place_enemies(maze, enemy_density=0.10, seed=None):
    """Place enemies (mark with value 2) on open cells randomly up to enemy_density*total."""
    n = maze.shape[0]
    free = [(i, j) for i in range(n) for j in range(n) if maze[i, j] == 1]
    k = int(len(free) * enemy_density)
    if seed is not None:
        random.seed(seed)
    enemies = random.sample(free, k) if k>0 else []
    for (x, y) in enemies:
        maze[x, y] = 2  # enemy mark
    return enemies

def random_start_goal(maze, seed=None):
    """Choose random distinct start & goal among free cells (free or enemy allowed? assignment: Satellite starts on free cell; enemies unknown locations - so start/goal must be free (1))."""
    n = maze.shape[0]
    free = [(i, j) for i in range(n) for j in range(n) if maze[i, j] == 1]
    if len(free) < 2:
        raise ValueError("Not enough free cells")
    if seed is not None:
        random.seed(seed)
    s, g = random.sample(free, 2)
    return s, g

def in_bounds(n, x, y):
    return 0 <= x < n and 0 <= y < n

def build_full_graph(maze):
    """
    Build graph of all free/enemy cells (value !=0) as adjacency dict:
      graph[(x,y)] = { (nx,ny): actionUsedToGetThere }
    This full graph will be used to compute corridors and junctions.
    """
    n = maze.shape[0]
    graph = {}
    for i in range(n):
        for j in range(n):
            if maze[i, j] != 0:  # not asteroid
                neigh = {}
                for a, (dx, dy) in ACTION_MOVES.items():
                    ni, nj = i + dx, j + dy
                    if in_bounds(n, ni, nj) and maze[ni, nj] != 0:
                        neigh[(ni, nj)] = a
                graph[(i, j)] = neigh
    return graph

def get_junction_graph(full_graph, start, goal):
    """
    Reduce full grid graph into junction graph:
    - Nodes are: start, goal, and any cell with degree != 2 (endpoints or junctions)
    - Edges are straight corridor paths with accumulated action sequence and lengths.
    Returns adjacency: {node: [(neighbor_node, path_cells, path_actions, path_cost), ...]}
    path_cells includes the neighbor node but not the source (useful for drawing).
    """
    nodes = set()
    for v, neigh in full_graph.items():
        deg = len(neigh)
        if deg != 2:
            nodes.add(v)
    nodes.add(start)
    nodes.add(goal)

    # helper to walk corridor from a node along direction until reaching a node or dead end
    visited_edges = set()
    junction_adj = defaultdict(list)
    for node in list(nodes):
        for neigh, action in full_graph[node].items():
            # traverse only if this directed edge not visited
            if (node, neigh) in visited_edges:
                continue
            # walk along corridor
            path_cells = [neigh]
            path_actions = [action]
            cur = neigh
            prev = node
            visited_edges.add((node, neigh))
            # continue until hit junction node or dead end
            while True:
                if cur in nodes:
                    # reached node (junction, start, or goal)
                    break
                # cur degree should be 2; continue forward
                nexts = [(p, a) for p, a in full_graph[cur].items() if p != prev]
                if not nexts:
                    # dead end
                    break
                nxt, act = nexts[0]
                prev, cur = cur, nxt
                path_cells.append(cur)
                path_actions.append(act)
                visited_edges.add((prev, cur))
            # compute path cost using action costs for the sequence
            cost = sum(ACTION_COST[a] for a in path_actions)
            junction_adj[node].append((cur, path_cells, path_actions, cost))
            junction_adj[cur].append((node, list(reversed(path_cells)), list(reversed(path_actions)), cost))
    return dict(junction_adj)
