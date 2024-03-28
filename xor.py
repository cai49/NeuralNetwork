"""
In the following example, the XOR problem is solved using the designed libraries
the Neural Network architecture has 2 input neurons, 1 hidden layer with 3 neurons, and 1 output neuron
"""


import dense
from activation_functions import *
from dense import Dense
from losses import mse, mse_prime
from network import train, predict
from serialization import serialize, deserialize
from utils import show

# Input vectors
X = np.reshape(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (4, 2, 1))
Y = np.reshape(np.array([[0], [1], [1], [0]]), (4, 1, 1))

# NN description
neural_network = [
    Dense(2, 6),
    Tanh(),
    Dense(6, 6),
    Tanh(),
    Dense(6, 1),
    Sigmoid()
]

# Training Parameters
epochs = 10000
learning_rate = 0.025

# Training
train(neural_network, mse, mse_prime, X, Y, epochs, learning_rate, verbose=False)

# Serialization aka saving the weights and biases to an external file
serialize(neural_network, "model.json")
deserialize(neural_network, "model.json")

show(neural_network)
