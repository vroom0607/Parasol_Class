class Node:
    def __init__(self, state, g=0, h=0, parent=None):
        self.state = state #state is tuple with row i, column j, time step t
        self.g = g
        self.h = h
        self.parent = parent

    #without __eq__, two nodes at different times will not be treated as equal since they are in different places in memory
    def __eq__(self, other): #if two nodes are equal
        return self.state == other.state

    def __hash__(self): #hash node so tuple can be looked up in closed() set
        return hash(self.state)