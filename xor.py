"""
In the following example, the XOR problem is solved using the designed libraries
the Neural Network architecture has 2 input neurons, 1 hidden layer with 3 neurons, and 1 output neuron
"""
import matplotlib.pyplot as plt, mpld3
from mpl_toolkits.mplot3d import Axes3D

from activation_functions import *
from dense import Dense
from losses import mse, mse_prime
from network import train, predict

# Input vectors
X = np.reshape(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (4, 2, 1))
Y = np.reshape(np.array([[0], [1], [1], [0]]), (4, 1, 1))

# NN description
neural_network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 2),
    Tanh(),
    Dense(2, 1),
    Tanh()
]

# Training Parameters
epochs = 10000
learning_rate = 0.01

# Training
train(neural_network, mse, mse_prime, X, Y, epochs, learning_rate, verbose=True)

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
