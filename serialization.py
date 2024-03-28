import json
from dense import Dense


def serialize(neural_network, filename):
    weights = []
    biases = []
    for layer in neural_network:
        if type(layer) is Dense:
            weights.append(layer.weights.tolist())
            biases.append(layer.bias.tolist())

    model = {'weights': weights, 'biases': biases}

    with open(filename, "w") as model_file:
        json.dump(model, model_file)


def deserialize(neural_network, filename, verbose=False):
    with open(filename, "r") as model_file:
        model = json.load(model_file)

        if verbose:
            print(model)

        j = 0
        for i, layer in enumerate(neural_network):
            if type(layer) is Dense:
                layer.weights = model['weights'][j]
                layer.bias = model['biases'][j]
                j += 1
