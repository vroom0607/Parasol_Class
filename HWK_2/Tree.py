import numpy as np
from Node import Node

class Tree:
    def __init__(self, start):
        self.root = Node(start)
        self.nodes = [self.root]

    #inserts a new node
    def insert(self, state, parent):
        new_node = Node(state, parent)
        self.nodes.append(new_node)
        return new_node

    #returns the nearest node to a given state(point)
    def nearest(self, state):
        nearest_node = self.nodes[0]
        min_dist = np.linalg.norm(nearest_node.state - state)

        for node in self.nodes:
            dist = np.linalg.norm(node.state - state)

            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        return nearest_node

    #finds the path by following parent pointers backwards
    def get_path(self, node):
        path = []

        while node is not None:
            path.append(node.state)
            node = node.parent

        return path[::-1] #reverses list