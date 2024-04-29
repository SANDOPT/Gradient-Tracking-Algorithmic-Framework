import numpy as np
import pandas as pd

"""Quadratic Problem
The methodoly follows  : A. Mokhtari, Q. Ling and A. Ribeiro, "Network Newton Distributed Optimization Methods," in IEEE Transactions on Signal Processing, vol. 65
"""


def Quadratic_problem_generator(p, xi, n, seed=10):
    s1 = []
    s2 = []
    v = []

    rng = np.random.default_rng(seed=seed)

    X_node = []
    y_node = []

    for i in range(xi + 1):
        s1.append(10 ** (-i))
        s1.append(10 ** (-i))
        s2.append(10 ** (i))
        s2.append(10 ** (i))

    for i in range(n):
        v1_ = list(rng.choice(s1, size=int(p / 2)))
        v2_ = list(rng.choice(s2, size=int(p / 2)))

        v = v + v1_ + v2_

    X = np.diag(v)
    y = rng.uniform(0, 1, n * p)

    sumX = np.zeros((p, p))
    sumy = np.zeros((p, 1))

    for it in range(n):

        X_node = X_node + [X[it * p : p * (it + 1), it * p : p * (it + 1)]]
        y_node = y_node + [y[it * p : p * (it + 1)][np.newaxis].T]
        sumX = sumX + X_node[-1]
        sumy = sumy + y_node[-1]

    return sumX, sumy, X_node, y_node


# Gradient for the Quadratic problem in the centralized setting
def Quadratic_central_gradient(A, b, x):

    return np.dot(A, x) + b


# Gradient for the Quadratic problem in the centralized setting
def Quadratic_decentralized_gradient(A_list, b_list, x):

    f = np.zeros(np.shape(x))

    for i in range(len(A_list)):

        f[i, :] = np.dot(A_list[i], x[i, :][np.newaxis].T).T + b_list[i].T

    return f


"""Logistic Regression Problem"""


def Logistic_regression_load_data(location):

    data = pd.read_csv(location).to_numpy().astype(int)
    X = data[:, 2:]
    Y = data[:, :2].T
    K = 2
    m, n = np.shape(X)

    return X.T, Y[:-1, :], m, n, K


# Split dataset to decentralized setting
def split_dataset(X, Y, m, n):

    s = int(m / n)

    X_list = []
    Y_list = []

    for i in range(n - 1):
        X_list.append(X[:, i * s : (i + 1) * s])
        Y_list.append(Y[:, i * s : (i + 1) * s])

    X_list.append(X[:, (n - 1) * s :])
    Y_list.append(Y[:, (n - 1) * s :])

    return X_list, Y_list


# Logistic Regression Gradient for cross entropy loss
def logistic_grad(X, Y, x):

    p, m = np.shape(X)
    K, _ = np.shape(Y)

    x = x.reshape((p, K))

    ET = np.exp(np.dot(x.T, X))
    temp = np.sum(ET, axis=0) + 1
    P = ET / temp

    g = np.dot(X, (P - Y).T) / m + 2 * x / m

    return g.reshape((p * K, 1))


# Loggistic regression decentralized gradient list
def logistic_grad_full(A_list, b_list, x):

    f = np.zeros(np.shape(x))

    for i in range(len(A_list)):

        f[i, :] = logistic_grad(A_list[i], b_list[i], x[i, :][np.newaxis].T).T

    return f
