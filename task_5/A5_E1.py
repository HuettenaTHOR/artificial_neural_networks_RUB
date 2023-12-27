import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



def poly_fit(random_seed, N, X, Y, ax_data, ax_error):
    mse_train = np.empty(len(N))
    mse_test = np.empty(len(N))

    # insert your codes below
    for n_index, n in enumerate(N):
        poly = PolynomialFeatures(degree=n)
        exp_X = poly.fit_transform(X)

        train_X, val_X, train_Y, val_Y = train_test_split(exp_X, Y, train_size=0.8, random_state=random_seed)

        ax_data.scatter(train_X[:, 1], train_Y, c="blue")

        lin_reg = LinearRegression(fit_intercept=False)
        reg = lin_reg.fit(train_X, train_Y)

        X_fit = np.linspace(X.min(), X.max(), 100).reshape((-1, 1))
        # print(x_fit)

        # print(reg.coef_)
        X_fit_poly = poly.fit_transform(X_fit)
        Y_fit = lin_reg.predict(X_fit_poly)
        ax_data.plot(X_fit, Y_fit)

        mse_train[n_index] = np.mean((lin_reg.predict(train_X) - train_Y)**2)
        mse_test[n_index] = np.mean((lin_reg.predict(val_X) - val_Y)**2)

    ax_error.plot(N, mse_train, 'o-', label='training set')
    ax_error.plot(N, mse_test, 'o-', label='test set')


data = np.load("05_model_selection_data.npy")
X = data[:, 0].reshape((-1, 1))
Y = data[:, 1]

N = [1, 3, 5, 7]
random_seeds = [0, 1, 2]

fig, ax = plt.subplots(2, len(random_seeds), figsize=(10, 5.5))
ax[0, 0].set_ylabel("Y")
ax[1, 0].set_ylabel("Mean Squared Error")

for col, random_seed in enumerate(random_seeds):
    ax[0, col].scatter(X, Y, color='gray')

    poly_fit(random_seed, N, X, Y, ax[0, col], ax[1, col])

    ax[0, col].set_title(f"random seed = {random_seed}")
    ax[0, col].set_xlabel("X")
    ax[0, col].set_ylim([-0.6, 2.1])
    ax[0, col].legend(loc='upper center')
    ax[1, col].set_xlabel("Polynomial degree")
    ax[1, col].set_xticks(N)
    ax[1, col].legend(loc='upper center')

fig.tight_layout()
plt.show()
