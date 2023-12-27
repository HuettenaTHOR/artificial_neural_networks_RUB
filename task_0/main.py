from matplotlib import pyplot as plt
import numpy as np


def cubic_function(x):
    return 300 * x * x


def add_random(x):
    factor = 25
    return x + factor * (np.random.rand() - 0.5)


def main():
    N = 50
    x = np.arange(-1, 1, 1 / N)
    print(len(x))
    y = [cubic_function(_x) for _x in x]
    y_sample = [add_random(cubic_function(_x)) for _x in x]
    plt.subplot(1, 3, 1)
    plt.plot(x, y)
    plt.title("distribution")

    plt.subplot(1, 3, 2)
    plt.plot(x, y_sample)
    plt.title("sample")

    plt.subplot(1, 3, 3)
    v = [0, 1]

    plt.plot(v, v)

    a = y_sample
    x_quantil = np.arange(0, 1, 0.01)
    y_quantils = [np.quantile((x, y_sample), q=_x_q) for _x_q in x_quantil[:50]]

    print(np.quantile(a, 0.5))
    plt.scatter(x_quantil[:50], y_quantils)
    plt.title("Q-Q-Plot")

    plt.show()


if __name__ == "__main__":
    main()
