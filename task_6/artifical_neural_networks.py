import numpy as np
from scipy.special import expit
from matplotlib import pyplot as plt

def approximation_function(x, v, bias, weight, N, c = .5):
    sum = 0

    for i in range(N):
        sum += v[i]*expit(weight[i]*x+bias[i])
    return sum

def main():
    x = np.arange(-2.5, 2.5, step=.1)
    b = np.array([-2])
    v = np.array([1.])
    N = 1
    weights = [1, 10, 100, 1000]

    for w in weights:
        y = [approximation_function(x, [v], [b], [w], N) for x in x]
        plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    main()
