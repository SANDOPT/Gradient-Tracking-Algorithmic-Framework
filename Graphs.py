import numpy as np


def W_full(n):

    return np.ones([n, n]) / n


def W_full_almost_1(n):

    W = W_full(n)
    W[-1, -1] = W[-1, -2] + W[-1, -1]
    W[-2, -2] = W[-1, -2] + W[-2, -2]
    W[-1, -2] = 0
    W[-2, -1] = 0

    return W


def W_full_almost_2(n):

    W = W_full(n)
    for i in range(n - 1):

        W[i, i] = W[i, i] + W[i + 1, i]
        W[i + 1, i + 1] = W[i + 1, i + 1] + W[i + 1, i]
        W[i + 1, i] = 0
        W[i, i + 1] = 0

    return W


def W_line(n, t):

    W = np.zeros([n, n])

    for i in range(n):

        if i == 0:
            W[i, i] = 1 - t
            W[i, i + 1] = t

        elif i == n - 1:
            W[i, i] = 1 - t
            W[i, i - 1] = t
        else:
            W[i, i] = 1 - 2 * t
            W[i, i + 1] = t
            W[i, i - 1] = t
    return W


def W_cyclic(n, t):
    W = np.zeros([n, n])
    for i in range(n):

        W[i, i] = 1 - 2 * t

        if i == 0:
            W[i, n - 1] = t
        else:
            W[i, i - 1] = t

        if i == n - 1:
            W[i, 0] = t
        else:
            W[i, i + 1] = t

    return W


def W_star(n, t):

    W = np.zeros([n, n])
    W[0, :] = t
    W[:, 0] = t
    W[0, 0] = 1 - (n - 1) * t

    for i in range(1, n):
        W[i, i] = 1 - t
    return W


def beta(W):

    return np.linalg.eigh(W)[0][-2]
