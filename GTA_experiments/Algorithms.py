import numpy as np
from tqdm import tqdm


def GTA1(x_0, W, iterations, alpha, grad_full, n_g, n_c, x_opt):

    W = np.linalg.matrix_power(W, n_c)

    temp1 = grad_full(x_0)

    y = temp1
    opt_error = np.zeros(iterations)
    con_x_error = np.zeros(iterations)

    for i in tqdm(range(iterations)):

        x_1 = np.dot(W, x_0) - alpha * y
        temp2 = grad_full(x_1)
        y = np.dot(W, y) + temp2 - temp1

        opt_error[i] = np.linalg.norm(np.mean(x_1, 0)[np.newaxis].T - x_opt)
        con_x_error[i] = np.linalg.norm(np.mean(x_1, 0) - x_1)

        x_0 = x_1
        temp1 = temp2

        for _ in range(n_g - 1):
            x_1 = x_0 - alpha * y
            temp2 = grad_full(x_1)
            y = y + temp2 - temp1
            x_0 = x_1
            temp1 = temp2

    return opt_error, con_x_error


def GTA2(x_0, W, iterations, alpha, grad_full, n_g, n_c, x_opt):

    W = np.linalg.matrix_power(W, n_c)
    x_0_1 = np.dot(W, x_0)
    temp1 = grad_full(x_0_1)
    y = temp1
    opt_error = np.zeros(iterations)
    con_x_error = np.zeros(iterations)

    for i in tqdm(range(iterations)):

        x_1 = x_0_1 - alpha * y
        x_1_1 = np.dot(W, x_1)
        temp2 = grad_full(x_1_1)
        y = np.dot(W, y) + temp2 - temp1
        opt_error[i] = np.linalg.norm(np.mean(x_1, 0)[np.newaxis].T - x_opt)
        con_x_error[i] = np.linalg.norm(np.mean(x_1_1, 0) - x_1_1)
        x_0_1 = x_1_1
        temp1 = temp2

        for _ in range(n_g - 1):

            x_1 = x_0_1 - alpha * y
            temp2 = grad_full(x_1)
            y = y + temp2 - temp1
            x_0_1 = x_1
            temp1 = temp2

    return opt_error, con_x_error


def GTA3(x_0, W, iterations, alpha, grad_full, n_g, n_c, x_opt):

    W = np.linalg.matrix_power(W, n_c)
    x_0_1 = np.dot(W, x_0)
    temp1 = grad_full(x_0_1)

    y = temp1
    opt_error = np.zeros(iterations)
    con_x_error = np.zeros(iterations)

    for i in tqdm(range(iterations)):

        x_1 = x_0_1 - alpha * y
        x_1_1 = np.dot(W, x_1)
        temp2 = grad_full(x_1_1)
        y = np.dot(W, y + temp2 - temp1)
        opt_error[i] = np.linalg.norm(np.mean(x_1, 0)[np.newaxis].T - x_opt)
        con_x_error[i] = np.linalg.norm(np.mean(x_1_1, 0) - x_1_1)
        x_0_1 = x_1_1
        temp1 = temp2

        for _ in range(n_g - 1):

            x_1 = x_0_1 - alpha * y
            temp2 = grad_full(x_1)
            y = y + temp2 - temp1
            x_0_1 = x_1
            temp1 = temp2

    return opt_error, con_x_error
