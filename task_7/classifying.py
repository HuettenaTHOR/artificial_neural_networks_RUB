import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from perceptron import Perceptron, PocketPerceptron

if not os.path.exists("figures"):
    os.makedirs("figures")

iris_dataset = datasets.load_iris()
X = iris_dataset.data[:, (0, 3)]
X = np.hstack((np.ones((X.shape[0], 1)), X))
Y = np.where(iris_dataset.target == 2, 1, -1)

for x, y in zip(X, Y):
    if y == 1:
        plt.scatter(x[1], x[2], c="red")
    else:
        plt.scatter(x[1], x[2], c="blue")


p = PocketPerceptron()
p.train(X, Y, 1, 700, 0)
x1 = np.array((X[:, 1].min(), X[:, 1].max()))
weights = p.weights_in_pocket
plt.plot(x1, -weights[0] / weights[2] - weights[1] / weights[2] * x1)
plt.title("Feature 0 and 3 with PP")
plt.show()
errors = p.training_errors
plt.title("Feature 0 and 3 with PP")
best_errors = p.smallest_training_errors
plt.plot(np.arange(0, len(errors)), errors)
plt.plot(np.arange(0, len(errors)), best_errors)
plt.show()
