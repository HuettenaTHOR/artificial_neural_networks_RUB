from matplotlib import pyplot as plt
import numpy as np


def f(x):
    return x**4 - 4 * x**2 + 4


x = np.linspace(-2, 2, 1000)
x = np.arange(-2, 2, 0.1)


def f_dash(x):
    return 4 * x**3 - 8 * x


def gradient_descent(ln_rate, n_iter, theta_init, derivation_function):
    theta = theta_init
    for i in range(int(n_iter)):
        theta = theta - ln_rate * derivation_function(theta)
    return theta


def main():
    x = np.arange(-2, 2, 1 / 40)
    y = [f(x) for x in x]

    plt.plot(x, y)
    plt.scatter(0, 4, c="green")
    plt.scatter(np.sqrt(2), 0, c="red")
    plt.scatter(-np.sqrt(2), 0, c="red")
    plt.show()

    ln_rates = np.arange(0.05, 0.15, 0.01)
    gradients = []
    for ln_rate in ln_rates:
        gradients.append(gradient_descent(ln_rate, 100, 1, f_dash))
    accuracies = [np.sqrt(2) - gradient for gradient in gradients]
    plt.scatter(ln_rates, accuracies)
    plt.show()


if __name__ == "__main__":
    main()
