import numpy as np

class Node:
    def __init__(self, state):
        self.state = state #(x,y), np array
        self.neighbors = []
        self.g = float("inf")
        self.parent = None
    
    def get_state(self):
        return self.state

    def get_neighbors(self):
         return self.neighbors