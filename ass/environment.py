
# environment.py
import math

class SatelliteAgent:
    def __init__(self, name, performance_init):
        self.name = name
        self.performance = performance_init  # float
        self.alive = True
        self.path_taken = []  # low-level cells visited
        self.node_path = []   # junction nodes visited (for high-level)
        self.current = None

    def encounter_enemy(self, enemy_power):
        """
        Apply enemy rule:
         - If enemy_power >= 2 * agent.performance => agent dies (captured)
         - Else agent activates defense, consumes 10% of current performance and survives
        Returns True if survives, False if dies.
        """
        if enemy_power >= 2 * self.performance:
            self.alive = False
            return False
        # defense uses 10% of current performance
        self.performance -= 0.10 * self.performance
        return True

class AsteroidEnvironment:
    def __init__(self, maze, enemies, enemy_powers, start, goal, junction_graph):
        """
        maze: numpy array with values:
            0 asteroid, 1 free, 2 enemy
        enemies: list of enemy coords (2s)
        enemy_powers: dict {(x,y): power}
        start, goal: tuples
        junction_graph: reduced graph
        """
        self.maze = maze.copy()
        self.enemies = set(enemies)
        self.enemy_powers = enemy_powers
        self.start = start
        self.goal = goal
        self.graph = junction_graph

    def traverse_low_level(self, agent, low_level_path):
        """
        low_level_path is list of cells to visit in order (assumed contiguous, free/enemy).
        Move agent cell-by-cell, handling enemies, adjusting performance, stopping if dead.
        Return (success_reached_goal_boolean, new_performance, alive)
        """
        for cell in low_level_path:
            if not agent.alive:
                break
            agent.current = cell
            agent.path_taken.append(cell)
            # check enemy
            if cell in self.enemies:
                power = self.enemy_powers[cell]
                survived = agent.encounter_enemy(power)
                if not survived:
                    # agent captured
                    return False, agent.performance, False
            # If agent reaches target cell coordinate, success may be considered by caller
            if cell == self.goal:
                return True, agent.performance, agent.alive
        # if path finished but not at goal, not success
        return False, agent.performance, agent.alive
