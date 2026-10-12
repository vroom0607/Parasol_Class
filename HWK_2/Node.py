import numpy as np

class Node:
    def __init__(self, state=np.array([0,0]), parent=None):
        self.state = state #(x,y), np array
        self.parent = parent

    def get_state(self):
        return self.state

    def get_parent(self):
        return self.parent