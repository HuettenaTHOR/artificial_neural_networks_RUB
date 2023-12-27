
from matplotlib import pyplot as plt

import numpy as np


def loss_function(theta):
    return theta**2


def loss_function_der(theta):
    return 2 * theta


def gradient_descent(ln_rate, n_iter, theta_init, derivation_function):
    thetas = []
    theta = theta_init
    thetas.append(theta)
    for i in range(int(n_iter)):
        theta = theta - ln_rate * derivation_function(theta)
        thetas.append(theta)
    return thetas


if __name__ == "__main__":
    x = np.arange(-2, 2, 0.1)

    ln_rates = np.arange(0.1, 1.1, 0.1)

    thetas = [gradient_descent(x, 100, 1, loss_function_der) for x in ln_rates]
    for pl_index in range(10):
        plt.subplot(2, 5, pl_index + 1)
        plt.plot(x, [loss_function(x) for x in x])
        plt.scatter(
            thetas[pl_index], [loss_function(x) for x in thetas[pl_index]], c="red"
        )
        plt.title(f"Learning rate: {round(ln_rates[pl_index], 1)}")
    plt.show()
