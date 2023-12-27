import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold


def poly_fit(X, Y, regression_model, lambdas, ax_error, ax_coefs, k=20):
    mse_train = np.empty((lambdas.size, k))
    mse_test = np.empty((lambdas.size, k))



    poly = PolynomialFeatures(degree=9)
    _X = poly.fit_transform(X)
    for i, lmbda in enumerate(lambdas):
        k_fold = KFold(n_splits=k, shuffle=True)

        coefs = np.empty((k, 10))

        for fold_index, (train_indices, test_indices) in enumerate(k_fold.split(_X)):
            reg = regression_model(fit_intercept=False, alpha=lmbda)
            reg.fit(_X[train_indices], Y[train_indices])
            mse_train[i][fold_index] = np.mean((reg.predict(_X[train_indices]) - Y[train_indices])**2)
            mse_test[i][fold_index] = np.mean((reg.predict(_X[test_indices]) - Y[test_indices])**2)
            coefs[fold_index] = reg.coef_.flatten()

        ax_coefs.plot(range(0, 10), np.mean(coefs, axis=0), 'o-', label=f"lambda: {lmbda}")



    # insert your codes below 
    ax_error.plot(lambdas, np.mean(mse_train, axis=1), 'o-', label='9th deg, train')
    ax_error.plot(lambdas, np.mean(mse_test, axis=1), 'o-', label='9th deg, test')


data = np.load("05_model_selection_data.npy")
X = data[:, 0].reshape((-1, 1))
Y = data[:, 1]

# polynomial of third degree for comparison
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)
k = 20
k_fold = KFold(n_splits=k, shuffle=True)
mse_train = np.empty(k)
mse_test = np.empty(k)
for fold_index, (train_indices, test_indices) in enumerate(k_fold.split(X_poly)):
    reg = LinearRegression(fit_intercept=False).fit(X_poly[train_indices], Y[train_indices])
    mse_train[fold_index] = np.mean((reg.predict(X_poly[train_indices]) - Y[train_indices]) ** 2)
    mse_test[fold_index] = np.mean((reg.predict(X_poly[test_indices]) - Y[test_indices]) ** 2)
mse_train_3 = np.mean(mse_train)
mse_test_3 = np.mean(mse_test)

lambdas = np.linspace(0.001, 0.03, 5)

fig, ax = plt.subplots(2, 2, figsize=(8, 6))

for col, regression_model in enumerate([Ridge, Lasso]):
    ax[0, col].set_title("Ridge Regression" if regression_model == Ridge else "Lasso Regression")
    ax[0, col].axhline(mse_train_3, linestyle='dashed', color='C0', label="3rd deg, train")
    ax[0, col].axhline(mse_test_3, linestyle='dashed', color='C1', label="3rd deg, test")
    ax[1, col].axhline(0, linestyle='dashed', color='lightgray')

    poly_fit(X, Y, regression_model, lambdas, ax[0, col], ax[1, col])

    ax[0, col].set_xlabel("Lambda")
    ax[1, col].set_xticks(range(0, 10))
    ax[1, col].set_xlabel("Polynomial degree")

ax[0, 0].set_ylabel("Mean Squared Error")
ax[1, 0].set_ylabel("Coefficient value")
ax[0, 1].legend(loc='upper left', bbox_to_anchor=(1.1, 1.05))
ax[1, 1].legend(loc='upper left', bbox_to_anchor=(1.1, 1.05))
fig.align_ylabels()
fig.tight_layout()
fig.savefig("regularization.pdf")
plt.show()
