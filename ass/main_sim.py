
# main_sim.py
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from asteroid_maze import make_maze, place_enemies, random_start_goal, build_full_graph, get_junction_graph
from agents_and_search import uniform_cost_search, iterative_deepening_search
from environment import SatelliteAgent, AsteroidEnvironment

import io, base64
from flask import Flask, send_file, jsonify, request, Response

# -------------------------
# Setup maze world
# -------------------------
N = 7
ASTEROID_DENSITY = 0.25
ENEMY_DENSITY = 0.10
seed = None  # set int for reproducible

maze = make_maze(N, ASTEROID_DENSITY, ENEMY_DENSITY, seed=seed)
enemies = place_enemies(maze, ENEMY_DENSITY, seed=seed)

# ensure start & goal are free cells (per earlier function)
start, goal = random_start_goal(maze, seed=seed)

# full graph and junction graph
full_graph = build_full_graph(maze)
junction_graph = get_junction_graph(full_graph, start, goal)

# -------------------------
# Enemy powers (10% - 40% of Nnodes)
# -------------------------
num_nodes = len(junction_graph)
enemy_powers = {}
for e in enemies:
    power = random.randint(int(0.10 * num_nodes), max(1, int(0.40 * num_nodes)))
    enemy_powers[e] = power

# -------------------------
# Agents initial performance
# -------------------------
initial_performance = 0.5 * num_nodes  # as assignment says 50% of number of nodes
agent1 = SatelliteAgent("UCS", initial_performance)
agent2 = SatelliteAgent("IDDFS", initial_performance)
agent1.current = start
agent2.current = start

# run searches to get planned low-level paths (both start same place)
node_path_ucs, low_path_ucs = uniform_cost_search(junction_graph, start, goal)
id_result = iterative_deepening_search(junction_graph, start, goal, max_depth=30)
if id_result is None:
    node_path_id = None
    low_path_id = None
else:
    node_path_id, low_path_id = id_result

# wrap environment
env = AsteroidEnvironment(maze, enemies, enemy_powers, start, goal, junction_graph)

# -------------------------
# Visualization helpers
# -------------------------
def draw_maze_state(maze, agents, start, goal):
    """
    Returns PNG bytes showing maze with:
      - black asteroids, white free, gold enemies
      - agents' current positions as colored markers and path traces
    """
    cmap = mcolors.ListedColormap(['black', 'white', 'gold'])  # 0,1,2
    plt.figure(figsize=(5,5))
    plt.imshow(maze, cmap=cmap, interpolation='none', vmin=0, vmax=2)
    # plot agents traces and current positions
    for ag, color in zip(agents, ['cyan', 'magenta']):
        if ag.path_taken:
            xs = [p[1] for p in ag.path_taken]
            ys = [p[0] for p in ag.path_taken]
            plt.plot(xs, ys, marker='o', color=color, markersize=4, linewidth=1, alpha=0.8)
        if ag.current is not None:
            plt.text(ag.current[1], ag.current[0], ag.name[0], color=color, fontsize=12, fontweight='bold',
                     ha='center', va='center')
    plt.text(start[1], start[0], 'S', color='green', fontsize=12, fontweight='bold', ha='center', va='center')
    plt.text(goal[1], goal[0], 'G', color='blue', fontsize=12, fontweight='bold', ha='center', va='center')
    plt.axis('off')

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf.getvalue()

# -------------------------
# Simple stepping function
# -------------------------
def step_agent_through(agent, low_path):
    """
    Move agent one low-level cell ahead along low_path (list of cells).
    Returns (reached_goal_boolean, alive_boolean)
    """
    if not low_path or not agent.alive:
        return False, agent.alive
    # move one cell
    next_cell = low_path.pop(0)
    succeeded, perf, alive = env.traverse_low_level(agent, [next_cell])
    return succeeded, alive

# We'll provide a small web-service to step simulation:
app = Flask(__name__)
state = {
    'low_path_ucs': low_path_ucs[:] if low_path_ucs else [],
    'low_path_id': low_path_id[:] if low_path_id else [],
    'agents': [agent1, agent2],
    'done': False
}

@app.route('/step', methods=['POST'])
def step():
    """Make one step for both agents (UCS then IDDFS) and return status."""
    if state['done']:
        return jsonify({"status":"done"}), 200
    agents = state['agents']
    # UCS moves
    success_ucs, alive_ucs = step_agent_through(agents[0], state['low_path_ucs'])
    # ID moves
    success_id, alive_id = step_agent_through(agents[1], state['low_path_id'])
    # check goal
    if agents[0].current == goal or agents[1].current == goal:
        state['done'] = True
    # build return info
    info = {
        "ucs_pos": agents[0].current,
        "ids_pos": agents[1].current,
        "ucs_alive": agents[0].alive,
        "ids_alive": agents[1].alive,
        "ucs_perf": agents[0].performance,
        "ids_perf": agents[1].performance
    }
    return jsonify(info)

@app.route('/maze.png')
def maze_png():
    png = draw_maze_state(maze, state['agents'], start, goal)
    return Response(png, mimetype='image/png')

@app.route('/status')
def status():
    return jsonify({
        "start": start,
        "goal": goal,
        "nodes": len(junction_graph),
        "ucs_path_len": len(low_path_ucs) if low_path_ucs else None,
        "id_path_len": len(low_path_id) if low_path_id else None,
        "enemy_positions": list(map(tuple, enemies)),
        "enemy_powers": {str(k): v for k, v in enemy_powers.items()}
    })

if __name__ == '__main__':
    # run flask for step-by-step control: visit /maze.png to see the map, POST to /step to advance by 1 low-level cell
    app.run(port=5000, debug=True)
