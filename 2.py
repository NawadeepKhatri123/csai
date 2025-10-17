
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import heapq, time

# ----------------------------------------------
# CONFIG
# ----------------------------------------------
ACTIONS = {0:(0,-1), 1:(-1,0), 2:(0,1), 3:(1,0)}   # left, up, right, down
ACTION_COST = {0:2,1:4,2:2,3:1}

# ----------------------------------------------
# MAZE GENERATION
# ----------------------------------------------
def generate_asteroid_maze(size=(7,7)):
    n = size[0]*size[1]
    maze = np.zeros(size,int)
    num_ast = int(n*0.25)
    num_enemy = int(n*0.10)

    idx = np.random.choice(n,num_ast,replace=False)
    maze[np.unravel_index(idx,size)] = 1

    free = np.where(maze==0)
    flat = np.ravel_multi_index(free,size)
    eidx = np.random.choice(flat,num_enemy,replace=False)
    maze[np.unravel_index(eidx,size)] = 2

    freecells = list(zip(*np.where(maze==0)))
    start, goal = np.random.choice(len(freecells),2,replace=False)
    start, goal = freecells[start], freecells[goal]
    maze[goal] = 3
    return maze, start, goal

# ----------------------------------------------
# ENTITIES
# ----------------------------------------------
class Enemy:
    def __init__(self, loc, power):
        self.loc, self.power = loc, power

class Satellite:
    def __init__(self, start, performance, maze, label):
        self.loc=start; self.perf=performance; self.maze=maze
        self.alive=True; self.path=[start]; self.label=label

    def defense(self):
        self.perf *= 0.9

def transition(state, act, maze):
    r,c = state; dr,dc = ACTIONS[act]; nr, nc = r+dr, c+dc
    if 0<=nr<maze.shape[0] and 0<=nc<maze.shape[1] and maze[nr,nc]!=1:
        return (nr,nc)
    return state

# ----------------------------------------------
# SEARCH METHODS
# ----------------------------------------------
def uniform_cost(start, goal, maze):
    frontier=[(0,start,[])]
    explored=set()
    while frontier:
        cost,state,path=heapq.heappop(frontier)
        if state==goal: return path
        if state in explored: continue
        explored.add(state)
        for a in ACTIONS:
            ns=transition(state,a,maze)
            if ns!=state:
                heapq.heappush(frontier,(cost+ACTION_COST[a],ns,path+[a]))
    return []

def depth_limited(state,goal,maze,path,limit,visited):
    if state==goal: return path
    if limit==0: return None
    visited.add(state)
    for a in ACTIONS:
        ns=transition(state,a,maze)
        if ns not in visited:
            res=depth_limited(ns,goal,maze,path+[a],limit-1,visited)
            if res is not None: return res
    return None

def iterative_deepening(start,goal,maze,maxdepth=50):
    for d in range(maxdepth):
        res=depth_limited(start,goal,maze,[],d,set())
        if res is not None: return res
    return []

# ----------------------------------------------
# VISUALIZATION
# ----------------------------------------------
def render_maze(maze, sat1, sat2):
    colormap = {0:"white",1:"black",2:"red",3:"green"}
    img = np.zeros((*maze.shape,3))
    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):
            img[r,c] = mcolors.to_rgb(colormap[maze[r,c]])
    plt.figure(figsize=(5,5))
    plt.imshow(img)
    plt.scatter(sat1.loc[1], sat1.loc[0], c='cyan', marker='o', label=sat1.label)
    plt.scatter(sat2.loc[1], sat2.loc[0], c='magenta', marker='x', label=sat2.label)
    plt.legend(loc="upper right")
    plt.xticks([]); plt.yticks([])
    st.pyplot(plt.gcf())
    plt.close()

# ----------------------------------------------
# STREAMLIT APP
# ----------------------------------------------
st.title("🛰️ Asteroid Maze — Satellite Agents")

if "initialized" not in st.session_state:
    maze, start, goal = generate_asteroid_maze()
    nodes = np.count_nonzero(maze != 1)
    enemies = [Enemy(tuple(pos), np.random.randint(int(0.1*nodes), int(0.4*nodes)))
               for pos in zip(*np.where(maze == 2))]
    s1 = Satellite(start, performance=0.5*nodes, maze=maze, label="UCS")
    s2 = Satellite(start, performance=0.5*nodes, maze=maze, label="ID-DFS")
    st.session_state.maze, st.session_state.start, st.session_state.goal = maze, start, goal
    st.session_state.enemies, st.session_state.s1, st.session_state.s2 = enemies, s1, s2
    st.session_state.step = 0
    st.session_state.path1 = uniform_cost(start, goal, maze)
    st.session_state.path2 = iterative_deepening(start, goal, maze)
    st.session_state.initialized = True

maze = st.session_state.maze
s1 = st.session_state.s1
s2 = st.session_state.s2
goal = st.session_state.goal
path1 = st.session_state.path1
path2 = st.session_state.path2
enemies = st.session_state.enemies

st.write(f"**Start:** {st.session_state.start} | **Goal (Earth):** {goal}")
st.write(f"Step: {st.session_state.step}")

render_maze(maze, s1, s2)

col1, col2 = st.columns(2)
if col1.button("Next Step ▶️"):
    step = st.session_state.step
    for sat, path in [(s1,path1),(s2,path2)]:
        if not sat.alive or step >= len(path): continue
        act = path[step]
        sat.loc = transition(sat.loc, act, maze)
        for e in enemies:
            if sat.loc == e.loc:
                if e.power >= 2*sat.perf:
                    sat.alive = False
                    st.warning(f"{sat.label} was captured by an enemy!")
                else:
                    sat.defense()
    st.session_state.step += 1
    st.rerun()

if col2.button("Reset 🔄"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

if s1.loc == goal or s2.loc == goal or not (s1.alive and s2.alive):
    st.success("Mission complete.")
    st.write(f"UCS — {'alive' if s1.alive else 'dead'} | Performance: {s1.perf:.1f}")
    st.write(f"IDDFS — {'alive' if s2.alive else 'dead'} | Performance: {s2.perf:.1f}")
    if s1.loc == goal and s2.loc != goal:
        st.subheader("🏆 Winner: UCS Agent")
    elif s2.loc == goal and s1.loc != goal:
        st.subheader("🏆 Winner: IDDFS Agent")
    elif s1.alive and s2.alive:
        st.subheader("🤝 Both reached Earth safely!")
    else:
        st.subheader("☠️ No survivors.")
