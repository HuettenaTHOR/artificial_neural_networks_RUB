import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def poly_fit(N, X, Y, k, axis):
    mse_train = np.empty((len(N), k))
    mse_test = np.empty((len(N), k))

    kfold = KFold(n_splits=k, shuffle=True)

    for n_index, n in enumerate(N):
        poly = PolynomialFeatures(degree=n)
        _X = poly.fit_transform(X)

        for fold_index, (train_indices, test_indices) in enumerate(kfold.split(_X)):
            reg = LinearRegression(fit_intercept=False)
            reg.fit(_X[train_indices], Y[train_indices])
            mse_train[n_index, fold_index] = np.mean((reg.predict(_X[train_indices]) - Y[train_indices]) ** 2)
            mse_test[n_index, fold_index] = np.mean((reg.predict(_X[test_indices]) - Y[test_indices]) ** 2)

    avg_mse_train = np.mean(mse_train, axis=1)
    avg_mse_test = np.mean(mse_test, axis=1)

    axis.plot(N, avg_mse_train, c="blue")
    axis.plot(N, avg_mse_test, c="red")


data = np.load("05_model_selection_data.npy")
X = data[:, 0].reshape((-1, 1))
Y = data[:, 1]

N = [1, 2, 3, 4, 5, 6, 7]
K = [2, 4, 5, 10, 20]

num_reps = 3
fig, ax = plt.subplots(num_reps, len(K), figsize=(10, 6))

for col, k in enumerate(K):
    ax[0, col].set_title(f"k = {k}")
    ax[-1, col].set_xlabel("Polynomial degree")

    for row in range(num_reps):
        poly_fit(N, X, Y, k, ax[row, col])

        ax[row, col].set_xticks(N)
        if col == 0:
            ax[row, 0].set_ylabel("Mean Squared Error")

ax[0, 0].legend(fontsize='small')
fig.align_ylabels()
fig.tight_layout()
fig.savefig("k-fold.pdf")
plt.show()
