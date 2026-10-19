import numpy as np
import heapq
import itertools
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
import random
import time

from Node import Node

MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]

#neighbors arent just physical neighbors but also neighbors in time
def build_neighbors(state, grid, reserved, swaps, parked, max_t): #reserved is a set of tuples of (i,j,t) so at every timestamp check if there is an agent there
    i, j, t = state
    new_t = t + 1
    size_i, size_j = grid.shape
    neighbors = []

    if new_t > max_t:
        return neighbors

    for di, dj in MOVES:
        new_i = i + di
        new_j = j + dj

        if not (0 <= new_i < size_i and 0 <= new_j < size_j) or grid[new_i, new_j] == 1:
            continue

        #vertex conflict with a higher-priority agent
        if (new_i, new_j, new_t) in reserved:
            continue

        #an agent who has reached its goal so it sits there
        if (new_i, new_j) in parked and new_t >= parked[(new_i, new_j)]:
            continue

        #swap conflict (not possible for a wait)
        if (di, dj) != (0, 0):
            if ((new_i, new_j), (i, j), t) in swaps:
                continue

        neighbors.append(Node((new_i, new_j, new_t)))

    return neighbors

def A_star(start_pos, dest_pos, grid, reserved, swaps, parked, max_t):
    counter = itertools.count()

    open = []
    closed = set() #set of tuples
    best_g = {} #holds the best g score found and is a dictionary of which state

    h = abs(start_pos[0] - dest_pos[0]) + abs(start_pos[1] - dest_pos[1])
    start_state = start_pos + (0,)
    start = Node(start_state, h=h)
    best_g[start] = 0

    # latest time any higher-priority agent touches the goal cell
    goal_busy_until = max((t for (i, j, t) in reserved if (i, j) == dest_pos), default=-1)

    heapq.heappush(open, (h, next(counter), start))

    while open:
        _,_,q = heapq.heappop(open)

        if q in closed: #check if node has already been checked, if so skip to next iteration (next node)
            continue

        if q.state[:2] == dest_pos and q.state[2] > goal_busy_until: 
            path = []
            while q is not None:
                path.append(q)
                q = q.parent
            return path[::-1]

        closed.add(q) #add q to closed list

        neighbors = build_neighbors(q.state, grid, reserved, swaps, parked, max_t)

        for neighbor in neighbors:
            g_new = q.g + 1

            if g_new < best_g.get(neighbor, float("inf")): #if neighbor has better g, if not in best_g then default to infinity
                best_g[neighbor] = g_new

                neighbor.parent = q
                neighbor.g = g_new
                
                h = abs(neighbor.state[0] - dest_pos[0]) + abs(neighbor.state[1] - dest_pos[1])
                f = g_new + h

                neighbor.h = h

                heapq.heappush(open, (f, next(counter), neighbor))

    return None

def update(path, reserved, swaps, parked):
    states = [n.state for n in path]
    state_prev = states[0]
    reserved.add(state_prev)

    for state in states[1:]:
        reserved.add(state)
        if state[:2] != state_prev[:2]:
            swap_value = (state_prev[:2], state[:2], state_prev[2])
            swaps.add(swap_value)
        state_prev = state

    reserved.add(states[-1])
    goal_i, goal_j, arrival = states[-1]
    parked[(goal_i, goal_j)] = arrival

def prioritized_search(grid, agents, goals, max_t, max_iterations):
    n = len(agents)
    priority_order = list(range(n))

    for iteration in range(max_iterations):
        random.shuffle(priority_order)

        reserved = set()
        swaps = set()
        parked = {}
        paths = {}
        failed = False

        for agent in priority_order:
            path = A_star(agents[agent][:2], goals[agent], grid, reserved, swaps, parked, max_t)

            if path is None:
                failed = True
                break

            update(path, reserved, swaps, parked)
            paths[agent] = [node.state for node in path]

        if not failed:
            return paths, iteration+1

    return None, max_iterations #if no solution found

def plot_paths(grid, agents, goals, paths, iterations, stats, save_path):
    n_rows, n_cols = grid.shape
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    fig, ax = plt.subplots(figsize=(14, 14))

    ax.imshow(grid, cmap="Greys", origin="upper", vmin=0, vmax=1.5)

    for agent, path in paths.items():
        color = colors[agent % len(colors)]
        offset = (agent - len(paths) / 2) * 0.03  # small shift so overlapping paths stay visible

        rows = [s[0] + offset for s in path]
        cols = [s[1] + offset for s in path]
        arrival = path[-1][2]

        ax.plot(cols, rows, "-", color=color, linewidth=1.8,
                label=f"Agent {agent} (arrives t={arrival})")

        # start is a square and goal is a star
        ax.plot(agents[agent][1], agents[agent][0], "s", color=color,
                markersize=9, markeredgecolor="black")
        ax.plot(goals[agent][1], goals[agent][0], "*", color=color,
                markersize=15, markeredgecolor="black")

    ax.set_xticks(np.arange(0, n_cols, 5))
    ax.set_yticks(np.arange(0, n_rows, 5))
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="lightgray", linewidth=0.5)
    ax.tick_params(which="minor", length=0)

    ax.set_xlabel("column j")
    ax.set_ylabel("row i")
    ax.set_title(
        f"Prioritized Planning on a {n_rows}x{n_cols} grid\n"
        f"Solved in {iterations} iteration(s) | "
        f"Compute time: {stats['runtime']:.3f} s | "
        f"Makespan: {stats['makespan']} steps | "
        f"Sum of costs: {stats['sum_costs']} steps"
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    seed = 4
    random.seed(seed)  # also controls the priority shuffling in prioritized_search

    n_rows, n_cols = 30, 30
    n_agents = 10
    obstacle_density = 0.15
    max_t = 150
    max_iterations = 200

    # 0 = free, 1 = obstacle
    rng = np.random.default_rng(seed)
    grid = (rng.random((n_rows, n_cols)) < obstacle_density).astype(int)

    # pick distinct free cells for every start and goal, as (row, col) tuples
    free_cells = [(int(i), int(j)) for i, j in zip(*np.where(grid == 0))]
    picks = rng.choice(len(free_cells), size=2 * n_agents, replace=False)
    agents = [free_cells[k] for k in picks[:n_agents]]
    goals = [free_cells[k] for k in picks[n_agents:]]

    start_time = time.perf_counter()
    paths, iterations = prioritized_search(grid, agents, goals, max_t, max_iterations)
    runtime = time.perf_counter() - start_time

    if paths is None:
        print(f"No solution found in {max_iterations} iterations ({runtime:.3f} s).")
        return

    costs = {a: len(p) - 1 for a, p in paths.items()}
    stats = {
        "runtime": runtime,
        "makespan": max(costs.values()),
        "sum_costs": sum(costs.values()),
    }

    print(f"Solution found in {iterations} iteration(s)")
    print(f"Compute time: {runtime:.3f} s")
    print(f"Makespan: {stats['makespan']} steps")
    print(f"Sum of costs: {stats['sum_costs']} steps")
    for agent in sorted(paths):
        print(f"Agent {agent}: {costs[agent]} steps")

    out_dir = Path(__file__).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "prioritized_search.png"

    plot_paths(grid, agents, goals, paths, iterations, stats, save_path)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    main()