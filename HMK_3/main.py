import numpy as np
import heapq
import itertools
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path

from Node import Node

def obstacle_check(state, radius, obstacles):
    dist = np.linalg.norm(state[None, :] - obstacles[:, :2], axis=1)

    return np.any(dist <= radius + obstacles[:, 2])

def random_configuration():
    x = np.random.uniform(0, 10)
    y = np.random.uniform(0, 10)

    state = np.array([x, y])

    return state

def edge_check(radius, obstacles, path):
    dist = np.linalg.norm(path[:, None, :] - obstacles[None, :, :2], axis=2)
    return not np.any(dist <= radius + obstacles[:, 2])

def local_planner(state1, state2, radius, obstacles):
    dist = np.linalg.norm(state2 - state1)
    n = max(2, int(np.ceil(dist / 0.01)) + 1) #make sure there are 2 points being checked or code will break

    edge = np.linspace(state1, state2, n)

    if not edge_check(radius, obstacles, edge):
        return None

    return edge

def A_star(start, dest): #start and dest are nodes
    counter = itertools.count()

    open = []
    closed = set()

    #score f is f = g + h
    h = np.linalg.norm(start.state - dest.state) #euclidean distance
    start.g = 0.0 #start node has g score of 0, g represents distance from starting node

    heapq.heappush(open, (h, next(counter), start)) #pushes a vector with f value, counter, and start node. f starts at h, counter prevents comparing to same node

    while open:
        _,_,q = heapq.heappop(open) #q is smallest item in the heap

        if q in closed: #check if node has already been checked, if so skip to next iteration (next node)
            continue

        if q is dest: #if node is dest add node to path and then return path backwards
            path = []
            while q is not None:
                path.append(q)
                q = q.parent
            return path[::-1]

        closed.add(q) #add q to closed list

        for neighbor in q.neighbors:
            cost = np.linalg.norm(neighbor.state - q.state)

            g_new = cost + q.g

            if g_new < neighbor.g:
                neighbor.g = g_new
                neighbor.parent = q

                h = np.linalg.norm(neighbor.state - dest.state)
                f = g_new + h

                heapq.heappush(open, (f, next(counter), neighbor))

    return None

def nearest_neighbors(node, nodes, check_radius):
    neighbors = []

    for i in nodes:
        if i is node:
            continue

        dist = np.linalg.norm(node.state - i.state)

        if dist <= check_radius:
            neighbors.append(i)

    return neighbors

def PRM(start, dest, radius, obstacles, max_iterations, check_radius):
    nodes = [start]

    for _ in range(max_iterations):
        sample = random_configuration()

        if obstacle_check(sample, radius, obstacles):
            continue

        nodes.append(Node(sample))

    nodes.append(dest)

    for node in nodes:
        node.neighbors = []
        node.g = np.inf
        node.parent = None
    
    for i, node in enumerate(nodes):
        neighbors = nearest_neighbors(node, nodes[i+1:], check_radius)

        for neighbor in neighbors:
            edge = local_planner(neighbor.state, node.state, radius, obstacles)

            if edge is not None:
                node.neighbors.append(neighbor)
                neighbor.neighbors.append(node)

    path = A_star(start, dest)
    return path, nodes

def plot_result(nodes, path, obstacles, radius, start, dest, title, filename="prm_result.png"):
    fig, ax = plt.subplots(figsize=(8, 8))

    # solid circle is the obstacle, dashed circle is inflated by the robot radius
    for x, y, r in obstacles:
        ax.add_patch(plt.Circle((x, y), r, color="tab:gray", alpha=0.6))
        ax.add_patch(plt.Circle((x, y), r + radius, fill=False, linestyle="--",
                                edgecolor="tab:gray", linewidth=0.8))

    # roadmap edges, drawn once per pair
    index = {id(n): i for i, n in enumerate(nodes)}
    segments = []
    for n in nodes:
        for m in n.neighbors:
            if index[id(n)] < index[id(m)]:
                segments.append([n.state, m.state])
    ax.add_collection(LineCollection(segments, colors="lightsteelblue",
                                     linewidths=0.4, alpha=0.7))

    pts = np.array([n.state for n in nodes])
    ax.scatter(pts[:, 0], pts[:, 1], s=4, color="steelblue", zorder=2)

    if path is not None:
        p = np.array([n.state for n in path])
        ax.plot(p[:, 0], p[:, 1], color="red", linewidth=2.5, zorder=3, label="Path")

    ax.scatter(*start.state, s=120, color="green", marker="o", zorder=4, label="Start")
    ax.scatter(*dest.state, s=120, color="blue", marker="*", zorder=4, label="Goal")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.savefig(filename, dpi=200, bbox_inches="tight")  # save before show()
    plt.show()

def main():
    seed = 0
    max_iterations = 1000
    check_radius = 1.0
    radius = 0.1  # robot radius

    np.random.seed(seed)

    # each row is (x, y, r)
    obstacles = np.array([
        [3.0, 3.0, 1.0],
        [5.0, 6.0, 1.2],
        [7.5, 3.0, 0.9],
        [2.5, 7.5, 0.8],
        [8.0, 8.0, 0.7],
    ])

    start = Node(np.array([0.5, 0.5]))
    dest = Node(np.array([9.5, 9.5]))

    for name, node in (("Start", start), ("Goal", dest)):
        if obstacle_check(node.state, radius, obstacles):
            print(f"{name} is inside an obstacle, pick a different position.")
            return

    path, nodes = PRM(start, dest, radius, obstacles, max_iterations, check_radius)

    # statistics
    num_nodes = len(nodes)
    sampled = num_nodes - 2  # excludes start and dest
    degrees = [len(n.neighbors) for n in nodes]
    num_edges = sum(degrees) // 2  # each edge is stored on both nodes
    straight = np.linalg.norm(dest.state - start.state)

    print("=" * 40)
    print("PRM RESULTS")
    print("=" * 40)
    print(f"Seed:                    {seed}")
    print(f"Samples attempted:       {max_iterations}")
    print(f"Samples rejected:        {max_iterations - sampled}")
    print(f"Nodes in roadmap:        {num_nodes}")
    print(f"Collision-free edges:    {num_edges}")
    print(f"Avg node degree:         {np.mean(degrees):.2f}")
    print(f"Max node degree:         {max(degrees)}")
    print(f"Isolated nodes:          {sum(d == 0 for d in degrees)}")
    print("-" * 40)

    if path is None:
        print("No path found.")
        title = "PRM: no path found"
    else:
        length = sum(np.linalg.norm(path[i + 1].state - path[i].state)
                     for i in range(len(path) - 1))
        print("Path found:              yes")
        print(f"Waypoints in path:       {len(path)}")
        print(f"Path length:             {length:.3f}")
        print(f"Straight-line distance:  {straight:.3f}")
        print(f"Path / straight-line:    {length / straight:.3f}")
        title = f"PRM path (length {length:.2f}, {len(path)} waypoints)"

    print("=" * 40)

    plot_result(nodes, path, obstacles, radius, start, dest, title,
                Path(__file__).parent / "prm.png")

if __name__ == "__main__":
    main()