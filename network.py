import matplotlib.pyplot as plt


def predict(network, input_data):
    output = input_data
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, loss, loss_prime, x_train, y_train, epochs=500, learning_rate=0.01, verbose=True):
    errors = []
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)
        errors.append(error)
    if verbose:

        plt.plot(range(epochs), errors, 'r')

        plt.title('Training Error')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Square Error')
        plt.show()
