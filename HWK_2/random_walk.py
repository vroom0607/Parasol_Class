import numpy as np
import matplotlib.pyplot as plt

def random_walk(start, dest):
    dx = 0
    dy = 0

    path = [start.copy()]

    while not np.array_equal(start, dest):
        dx = np.random.randint(-1,2)
        dy = np.random.randint(-1,2)

        start[0] += dx
        start[1] += dy

        path.append(start.copy())

    return path

def main():
    start = np.array([0,0])
    dest = np.array([10, 14])

    path = random_walk(start, dest)
    path = np.array(path)

    plt.plot(path[:, 0], path[:, 1], lw=0.5)
    plt.scatter(path[0, 0], path[0, 1], label="Start")
    plt.scatter(path[-1, 0], path[-1, 1], label="Destination")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()