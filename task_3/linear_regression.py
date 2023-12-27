import numpy as np
from matplotlib import pyplot as plt

def gt(x):
    return 4*x + 5 + np.random.uniform(-0.5, 0.5)

def main():
    x = np.linspace(-3, 3, 100)
    y = [gt(x) for x in x]
    print("linear regression")
    N = len(x)
    cov_x_y = np.sum(x*y) - (1/N)*(np.sum(x)*np.sum(y))
    var_x = (np.sum(x**2)-1/N*(np.sum(x)**2))
    m = cov_x_y / var_x
    print("m: ", m)
    b = np.mean(y) - m * np.mean(x)
    print("b: ", b)

    print("multiple linear regressen")

    size = 10
    bias = np.ones(N)
    X = np.vstack((bias, x))
    print(X)

    # print(x)
    x_t = np.transpose([x])
    # print(np.transpose([x]))
    # print(np.linalg.det(np.transpose([x])* [x]))
    # theta_head = np.linalg.inv([x]* np.transpose([x]))*np.transpose(x)*y
    # print(theta_head)

    plt.plot(x, y)
    plt.show()



if __name__ == "__main__":
    main()