import numpy as np
from Tree import Tree

class RRT:
    def __init__(self, start, dest, radius, obstacles, step_size, max_iterations):
        self.start = Tree(start)
        self.dest = dest #np array
        self.radius = radius #robot radius
        self.obstacles = obstacles #circles (x,y,r)
        self.step_size = step_size
        self.max_iterations = max_iterations

    #checks if robot of radius is colliding with any obstacles, returns bool
    def obstacle_check(self, state):
        dist = np.linalg.norm(state[None, :] - self.obstacles[:, :2], axis=1)

        return np.any(dist <= self.radius + self.obstacles[:, 2])

    #checks if straight line path between 2 points is collision free
    def path_check(self, point1, point2):
        distance = np.linalg.norm(point2 - point1)
        n = int(np.ceil(distance / 0.01))

        for state in np.linspace(point1, point2, n):
            if self.obstacle_check(state):
                return False

        return True

    #random point, np array
    def random_point(self):
        x = np.random.uniform(0, 10)
        y = np.random.uniform(0, 10)

        return np.array([x, y])

    #takes nearest node and random point and steps towards point
    def steer(self, nearest, random_state):
        dist = np.linalg.norm(random_state - nearest.state)

        if dist <= self.step_size:
            return random_state

        #start point + unit vector * step size
        return nearest.state + ((random_state - nearest.state) / dist) * self.step_size

    #finds plan
    def plan(self):
        for _ in range(self.max_iterations):
            random_state = self.random_point()

            nearest = self.start.nearest(random_state)

            new_state = self.steer(nearest, random_state)

            if not self.path_check(nearest.state, new_state):
                continue

            new_node = self.start.insert(new_state, nearest)

            if np.linalg.norm(new_state - self.dest) <= self.step_size:
                if self.path_check(new_state, self.dest):
                    goal_node = self.start.insert(self.dest, new_node)
                    return self.start.get_path(goal_node)

        return None