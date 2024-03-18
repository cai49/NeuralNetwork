"""
This is the main file for the custom Neural Network library

In the following example, the XOR problem is solved using the designed libraries
the Neural Network architecture has 2 input neurons, 1 hidden layer with 3 neurons, and 1 output neuron
"""
from dense import Dense
from activation_functions import *
from losses import mse, mse_prime
import numpy as np

# Input vectors
X = np.reshape(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (4, 2, 1))
Y = np.reshape(np.array([[0], [1], [1], [0]]), (4, 1, 1))

# NN description
neural_network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

# Training Parameters
epochs = 10000
learning_rate = 0.01

# Training
for e in range(epochs):
    error = 0

    for x, y in zip(X, Y):
        output = x
        for layer in neural_network:
            output = layer.forward(output)
        # Error
        error += mse(y, output)

        # Backwards
        grad = mse_prime(y, output)
        for layer in reversed(neural_network):
            grad = layer.backward(grad, learning_rate)

    error /= len(X)
    print('%d/%d, error=%f' % (e + 1, epochs, error))
