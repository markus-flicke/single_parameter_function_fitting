import numpy as np

def get_parabola():
    x = np.asarray([i for i in range(-50,51)])
    y = np.asarray([i**2.0 for i in x])

    x = x.reshape((len(x), 1))
    y = y.reshape((len(y), 1))

    return x, y


def get_polynomial(degree = 2.0):
    x = np.asarray([i for i in range(-50,51)])
    y = np.asarray([i**degree for i in x])

    x = x.reshape((len(x), 1))
    y = y.reshape((len(y), 1))

    return x, y


def get_noisy_polynomial(degree = 2.0):
    """
    adds uniform noise to a polynoimal
    """
    x = np.asarray([i for i in range(-50,51)])
    y = np.asarray([i**degree * (1 + np.random.normal()) for i in x])

    x = x.reshape((len(x), 1))
    y = y.reshape((len(y), 1))

    return x, y
