import numpy as np


def mse(y_curr, y_pred):
    return np.mean(np.square(y_curr - y_pred))


def mse_prime(y_curr, y_pred):
    return 2 * (y_pred - y_curr) / np.size(y_curr)
