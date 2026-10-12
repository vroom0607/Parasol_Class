import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

from RRT import RRT


def path_length(path):
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def path_is_valid(path, obstacles, robot_radius, resolution=0.05):
    """Densely sample each segment and check clearance against every obstacle."""
    for a, b in zip(path[:-1], path[1:]):
        n = max(2, int(np.linalg.norm(b - a) / resolution))
        for t in np.linspace(0.0, 1.0, n):
            p = a + t * (b - a)
            d = np.linalg.norm(obstacles[:, :2] - p, axis=1) - obstacles[:, 2]
            if np.any(d < robot_radius):
                return False
    return True


def run_once(start, dest, robot_radius, obstacles, seed,
             step_size=0.5, max_iterations=5000):
    np.random.seed(seed)
    rrt = RRT(start, dest, robot_radius, obstacles,
              step_size=step_size, max_iterations=max_iterations)

    t0 = time.perf_counter()
    path = rrt.plan()
    runtime = time.perf_counter() - t0

    result = {
        "seed": seed,
        "success": path is not None,
        "runtime": runtime,
        "tree_size": len(rrt.start.nodes),
        "rrt": rrt,
        "path": None,
    }
    if path is not None:
        path = np.array(path)
        result["path"] = path
        result["length"] = path_length(path)
        result["waypoints"] = len(path)
        result["valid"] = path_is_valid(path, obstacles, robot_radius)
    return result


def benchmark(start, dest, robot_radius, obstacles, n_trials=50):
    results = [run_once(start, dest, robot_radius, obstacles, seed=s)
               for s in range(n_trials)]
    ok = [r for r in results if r["success"]]
    times = np.array([r["runtime"] for r in results])

    print(f"\n--- Benchmark over {n_trials} seeds ---")
    print(f"Success rate : {len(ok)}/{n_trials} ({100 * len(ok) / n_trials:.0f}%)")
    print(f"Runtime (s)  : mean {times.mean():.4f}, median {np.median(times):.4f}, "
          f"std {times.std():.4f}, max {times.max():.4f}")
    if ok:
        lengths = np.array([r["length"] for r in ok])
        sizes = np.array([r["tree_size"] for r in ok])
        straight = np.linalg.norm(dest - start)
        print(f"Path length  : mean {lengths.mean():.2f}, min {lengths.min():.2f}, "
              f"std {lengths.std():.2f}")
        print(f"Optimality   : mean length is {lengths.mean() / straight:.2f}x "
              f"the straight-line distance ({straight:.2f})")
        print(f"Tree size    : mean {sizes.mean():.0f} nodes")
        print(f"All paths collision-free: {all(r['valid'] for r in ok)}")
    return results


def plot(result, start, dest, robot_radius, obstacles, output_path):
    rrt, path = result["rrt"], result["path"]
    fig, ax = plt.subplots(figsize=(7, 7))

    for x, y, r in obstacles:
        ax.add_patch(Circle((x, y), r, color="gray", alpha=0.8))
        ax.add_patch(Circle((x, y), r + robot_radius, fill=False,
                            linestyle="--", edgecolor="gray"))

    for node in rrt.start.nodes:
        if node.parent is not None:
            ax.plot([node.parent.state[0], node.state[0]],
                    [node.parent.state[1], node.state[1]],
                    color="lightblue", linewidth=0.8)

    if path is not None:
        ax.plot(path[:, 0], path[:, 1], color="red", linewidth=2.5, label="Path")

    ax.plot(*start, "go", markersize=10, label="Start")
    ax.plot(*dest, "r*", markersize=15, label="Goal")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    status = f"{result['runtime'] * 1000:.1f} ms, {result['tree_size']} nodes"
    ax.set_title(f"RRT Path Planning ({status})")
    ax.legend(loc="upper left")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def main():
    start = np.array([1.0, 1.0])
    dest = np.array([9.0, 9.0])
    robot_radius = 0.3
    obstacles = np.array([
        [3.0, 3.0, 1.0],
        [5.0, 6.0, 1.2],
        [7.5, 3.5, 0.9],
        [2.5, 7.5, 0.8],
        [6.5, 8.5, 0.7],
    ])

    # Single seeded run (this is the one that gets plotted)
    result = run_once(start, dest, robot_radius, obstacles, seed=0)
    print(f"Runtime      : {result['runtime'] * 1000:.2f} ms")
    print(f"Tree size    : {result['tree_size']} nodes")
    if result["success"]:
        print(f"Path found   : {result['waypoints']} waypoints, "
              f"length {result['length']:.2f}")
        print(f"Collision-free: {result['valid']}")
    else:
        print("No path found within max iterations")

    # Statistics across many seeds
    benchmark(start, dest, robot_radius, obstacles, n_trials=50)

    plot(result, start, dest, robot_radius, obstacles,
         Path(__file__).parent / "rrt_plot.png")


if __name__ == "__main__":
    main()