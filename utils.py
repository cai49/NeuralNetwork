from network import train, predict
import numpy as np
import matplotlib.pyplot as plt


def show(neural_network):
    # Boundary Plot
    points = []
    for x in np.linspace(0, 1, 20):
        for y in np.linspace(0, 1, 20):
            z = predict(neural_network, [[x], [y]])
            points.append([x, y, z[0, 0]])

    points = np.array(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")

    ax.set_title("3D plot")
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')

    plt.show()
