#Intermediate

import numpy as np

# arm_cspace is np array of joints with l,theta so shape (N, D=2)
def forward_kinematics(arm_cspace):
    L = arm_cspace[:, 0] #array of lengths
    theta = arm_cspace[:, 1] #array of angles

    # num_joints = arm_cspace.shape[0]

    # arm_workspace = np.zeros((num_joints, 4)) # includes x,y which is the endpoints of the link
    # total_theta = 0

    # for i in range(num_joints):
    #     arm_workspace[i, 2] = L[i]
    #     arm_workspace[i, 3] = theta[i]
        
    #     total_theta += theta[i]

    #     dx = L[i] * np.cos(total_theta)
    #     dy = L[i] * np.sin(total_theta)

    #     if i == 0:
    #         arm_workspace[i, 0] = dx
    #         arm_workspace[i, 1] = dy
    #     else:
    #         prev_x = arm_workspace[i-1, 0]
    #         prev_y = arm_workspace[i-1, 1]
    #         arm_workspace[i, 0] = prev_x + dx
    #         arm_workspace[i, 1] = prev_y + dy

    total_theta = np.cumsum(theta)

    dx = L * np.cos(total_theta)
    dy = L * np.sin(total_theta)

    x = np.cumsum(dx)
    y = np.cumsum(dy)

    arm_workspace = np.column_stack((x, y, L, theta))

    return arm_workspace

def capsule_representation(arm_workspace, width):
    # num_joints = arm_workspace.shape[0]

    # arm_capsules = np.zeros((num_joints, 5)) #x,y,x_end,y_end,r

    # for i in range(num_joints):
    #     x2 = arm_workspace[i, 0] #endpoints
    #     y2 = arm_workspace[i, 1]

    #     if i == 0:
    #         x1 = 0
    #         y1 = 0
    #     else:
    #         x1 = arm_workspace[i-1, 0]
    #         y1 = arm_workspace[i-1, 1]

    #     arm_capsules[i, 0] = x1
    #     arm_capsules[i, 1] = y1
    #     arm_capsules[i, 2] = x2
    #     arm_capsules[i, 3] = y2
    #     arm_capsules[i, 4] = width / 2

    # return arm_capsules

    x2 = arm_workspace[:, 0]
    y2 = arm_workspace[:, 1]

    x1 = np.concatenate(([0], x2[:-1])) #add 0 to endpoints and get rid of last index to get startpoints
    y1 = np.concatenate(([0], y2[:-1]))

    radius = np.full(len(arm_workspace), width / 2) #create radius array with same shape

    arm_capsules = np.column_stack((x1, y1, x2, y2, radius)) #add radius

    return arm_capsules

def collision_check(arm_capsules, obstacles):
    #arm coords, start and endpoints
    x1 = arm_capsules[:, 0]
    y1 = arm_capsules[:, 1]
    x2 = arm_capsules[:, 2]
    y2 = arm_capsules[:, 3]

    #radius of capsule
    r_capsule = arm_capsules[:, 4]

    #obstacle coords
    px = obstacles[:, 0]
    py = obstacles[:, 1]
    r_circle = obstacles[:, 2]

    #broadcasting
    #shape is (N, 1)
    x1 = x1[:, None]
    y1 = y1[:, None]
    x2 = x2[:, None]
    y2 = y2[:, None]
    r_capsule = r_capsule[:, None]

    #shape is (1, M)
    px = px[None, :]
    py = py[None, :]
    r_circle = r_circle[None, :]

    #(dx, dy) is direction of capsule
    dx = x2 - x1
    dy = y2 - y1

    #length of capsule (squared)
    length_squared = dx**2 + dy**2

    #dot product to find how far along capsule obstacle is, fraction from 0 to 1
    t = ((px - x1) * dx + (py - y1) * dy) / length_squared

    #keeps t between 0 and 1, so obstacle is not past the starting/enpoint of capsule
    t = np.clip(t, 0, 1)

    #the closest point (capsule_x, capsule_y) on the capsule segment to obstacle center
    x_closest = x1 + t * dx
    y_closest = y1 + t * dy

    #distance between obstacle center and closest point on capsule segment
    distance_squared = ((px - x_closest)**2 + (py - y_closest)**2)

    #if distance is less than the radii then its collision
    collision = distance_squared <= (r_capsule + r_circle)**2

    #boolean array
    return collision

def main():

    # Several arm configurations
    # Each row is one configuration
    # Each configuration has 3 joints
    # Each joint is [length, angle]
    configurations = np.array([
        [
            [2.0, 0.0],
            [1.5, 0.0],
            [1.0, 0.0]
        ],

        [
            [2.0, 0.0],
            [1.5, np.pi / 2],
            [1.0, 0.0]
        ],

        [
            [2.0, np.pi / 4],
            [1.5, -np.pi / 4],
            [1.0, np.pi / 2]
        ],

        [
            [2.0, np.pi / 2],
            [1.5, 0.0],
            [1.0, -np.pi / 2]
        ]
    ])

    # Obstacles: [x, y, radius]
    obstacles = np.array([
        [1.0, 0.0, 0.3],
        [2.0, 1.0, 0.3],
        [0.0, 2.0, 0.3]
    ])

    width = 0.2

    # Check every configuration
    for i, arm_cspace in enumerate(configurations):

        print(f"\nConfiguration {i}:")

        # Forward kinematics
        arm_workspace = forward_kinematics(arm_cspace)

        print("Workspace:")
        print(arm_workspace)

        # Convert links to capsules
        arm_capsules = capsule_representation(
            arm_workspace,
            width
        )

        print("Capsules:")
        print(arm_capsules)

        # Collision checking
        collisions = collision_check(
            arm_capsules,
            obstacles
        )

        print("Collision matrix:")
        print(collisions)

        # Does this configuration have ANY collision?
        print("Configuration in collision:", np.any(collisions))


if __name__ == "__main__":
    main()