#Novice

import numpy as np

#batch_cirlcles1 is all robot configurations
#ahape = (B, N, D)
def collision_check(batch_circles1, circles2):
    num_dims = circles2.shape[-1] - 1
    c1 = batch_circles1[..., :num_dims] #B,N,D-1
    c2 = circles2[:, :num_dims] #M,D-1
    r1 = batch_circles1[..., num_dims]
    r2 = circles2[:, num_dims]

    dist = np.sum((c1[:, :, None, :] - c2[None, None, :, :])**2, axis=-1) #B,N,M
    radii = (r1[:, :, None] + r2[None, None, :])**2

    return np.any(dist <= radii, axis=(1, 2)) #bool array

def main():

    # --------------------------------------------------
    # Robot configurations
    # --------------------------------------------------
    #
    # B = 3 configurations
    # N = 2 robot circles
    # D = 3 -> x, y, radius
    #
    robot = np.array([
        # Configuration 1
        [
            [0, 0, 1],
            [2, 0, 1]
        ],

        # Configuration 2
        [
            [4, 0, 1],
            [6, 0, 1]
        ],

        # Configuration 3
        [
            [10, 0, 1],
            [12, 0, 1]
        ]
    ])

    # --------------------------------------------------
    # Obstacles
    # --------------------------------------------------
    #
    # M = 2 obstacle circles
    # D = 3 -> x, y, radius
    #
    obstacles = np.array([
        [5, 0, 1],
        [20, 0, 1]
    ])

    # Check each configuration
    collisions = collision_check(robot, obstacles)

    print("Collision results:")
    print(collisions)


if __name__ == "__main__":
    main()