
from matplotlib import colors as mcolors  # add this import at the top of your file
import numpy as np
import heapq, time, matplotlib.pyplot as plt
np.random.seed(1)

ACTIONS = {0:(0,-1), 1:(-1,0), 2:(0,1), 3:(1,0)}   # left, up, right, down
ACTION_COST = {0:2,1:4,2:2,3:1}

def generate_asteroid_maze(size=(7,7)):
    n = size[0]*size[1]
    maze = np.zeros(size,int)
    num_ast = int(n*0.25)
    num_enemy = int(n*0.10)
    # place asteroids
    idx = np.random.choice(n,num_ast,replace=False)
    maze[np.unravel_index(idx,size)] = 1
    # place enemies
    free = np.where(maze==0)
    flat = np.ravel_multi_index(free,size)
    eidx = np.random.choice(flat,num_enemy,replace=False)
    maze[np.unravel_index(eidx,size)] = 2
    # start & goal
    freecells = list(zip(*np.where(maze==0)))
    start, goal = np.random.choice(len(freecells),2,replace=False)
    start, goal = freecells[start], freecells[goal]
    maze[goal]=3
    return maze, start, goal

class Enemy:
    def __init__(self, loc, power): self.loc, self.power = loc, power

class Satellite:
    def __init__(self, start, performance, maze, label):
        self.loc=start; self.perf=performance; self.maze=maze
        self.alive=True; self.path=[start]; self.label=label

    def defense(self):
        self.perf*=0.9

def transition(state, act, maze):
    r,c = state; dr,dc = ACTIONS[act]; nr, nc = r+dr, c+dc
    if 0<=nr<maze.shape[0] and 0<=nc<maze.shape[1] and maze[nr,nc]!=1:
        return (nr,nc)
    return state

# ---------- Uniform Cost Search ----------
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

# ---------- Iterative Deepening ----------
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

# ---------- Simulation ----------
def visualize(maze, sat1, sat2):
    colormap = {0: "white", 1: "black", 2: "red", 3: "green"}
    img = np.zeros((*maze.shape, 3))

    for r in range(maze.shape[0]):
        for c in range(maze.shape[1]):
            img[r, c] = mcolors.to_rgb(colormap[maze[r, c]])

    plt.imshow(img)
    plt.scatter(sat1.loc[1], sat1.loc[0], c='cyan', marker='o', label=sat1.label)
    plt.scatter(sat2.loc[1], sat2.loc[0], c='magenta', marker='x', label=sat2.label)
    plt.legend()
    plt.pause(0.3)
    plt.clf()

def simulate():
    maze,start,goal=generate_asteroid_maze()
    nodes=np.count_nonzero(maze!=1)
    enemies=[Enemy(tuple(pos),np.random.randint(int(0.1*nodes),int(0.4*nodes))) 
             for pos in zip(*np.where(maze==2))]
    s1=Satellite(start,performance=0.5*nodes,maze=maze,label="UCS")
    s2=Satellite(start,performance=0.5*nodes,maze=maze,label="ID-DFS")
    print("Start:",start,"Goal:",goal)
    path1=uniform_cost(start,goal,maze)
    path2=iterative_deepening(start,goal,maze)
    plt.ion()
    for step in range(max(len(path1),len(path2))):
        for sat,path in [(s1,path1),(s2,path2)]:
            if not sat.alive or step>=len(path): continue
            act=path[step]
            sat.loc=transition(sat.loc,act,maze)
            # enemy encounter
            for e in enemies:
                if sat.loc==e.loc:
                    if e.power>=2*sat.perf: sat.alive=False; print(sat.label,"captured!")
                    else: sat.defense()
        visualize(maze,s1,s2)
        if (s1.loc==goal or not s1.alive) and (s2.loc==goal or not s2.alive): break
    plt.ioff()
    print(f"Result UCS: {'alive' if s1.alive else 'dead'} perf={s1.perf:.1f}")
    print(f"Result IDDFS: {'alive' if s2.alive else 'dead'} perf={s2.perf:.1f}")
    if s1.loc==goal and s2.loc!=goal: print("Winner: UCS")
    elif s2.loc==goal and s1.loc!=goal: print("Winner: IDDFS")
    else: print("Draw")

if __name__=="__main__":
    simulate()
