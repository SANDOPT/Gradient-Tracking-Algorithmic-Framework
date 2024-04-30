import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import sys

sys.path.append("../")
import Graphs
import problems
import Algorithms


n = 16  # number of nodes
n_c = 5  # number of communication steps
p = 0.1  #  probability of computation steps
iterations = int(1e4)  # number of iterations
seed = 10  # seed for algorithm

"""Decentralized Graphs"""
# W = Graphs.W_full(n)
W = Graphs.W_star(n, 0.05)
# W = Graphs.W_cyclic(n, 0.05)
# W = Graphs.W_line(n, 0.05)
# W = Graphs.W_full_almost_2(n)
# W = Graphs.W_full_almost_1(n)

"""Logistic regression problem definition"""
problem_name = "Logistic_regression_mushroom"
location = "../Datasets/mushroom_processed.csv"
A, b, m, d, K = problems.Logistic_regression_load_data(location)
A_list, b_list = problems.split_dataset(A, b, m, n)
x_init = np.zeros([n, d * (K - 1)])
grad_full = lambda x: problems.logistic_grad_full(A_list=A_list, b_list=b_list, x=x)

# Running gradient descent for optimal solution
x_opt = np.zeros([(K - 1) * d, 1])
g = np.mean(problems.logistic_grad_full(A_list, b_list, np.tile(x_opt.T, (n, 1))), 0)
norm = np.linalg.norm(g)
while norm > 1e-10:
    x_opt = x_opt - g[np.newaxis].T
    g = np.mean(
        problems.logistic_grad_full(A_list, b_list, np.tile(x_opt.T, (n, 1))), 0
    )
    norm = np.linalg.norm(g)

"""Uncomment to run RGTA algorithms"""
# The step size (alpha) must be changed according to the problem and graph selected
"""Uncomment for RGTA1"""
# opt_error, con_error = Algorithms.RGTA1(
#     x_0=x_init,
#     W=W,
#     alpha=1e-3,
#     iterations=iterations,
#     grad_full=grad_full,
#     p = p,
#     n_c=n_c,
#     x_opt=x_opt,
#     seed=seed,
# )
# name = f"RGTA-1({n_c}, {p})"
"""Uncomment for RGTA2"""
# opt_error, con_error = Algorithms.RGTA2(
#     x_0=x_init,
#     W=W,
#     alpha=1e-3,
#     iterations=iterations,
#     grad_full=grad_full,
#     p = p,
#     n_c=n_c,
#     x_opt=x_opt,
#     seed=seed,
# )
# name = f"RGTA-2({n_c}, {p})"
"""Uncomment for RGTA3"""
opt_error, con_error = Algorithms.RGTA3(
    x_0=x_init,
    W=W,
    alpha=1e-3,
    iterations=iterations,
    grad_full=grad_full,
    p=p,
    n_c=n_c,
    x_opt=x_opt,
    seed=seed,
)
name = f"RGTA-3({n_c}, {p})"

"""Plotting code"""
fig, ax = plt.subplots(2, 2, figsize=(15, 7))
fig.suptitle(problem_name)
ax[0, 0].set_ylabel("Optimization Error")
ax[1, 0].set_ylabel("Consensus Error")

ax[0, 0].set_yscale("log")
ax[0, 1].set_yscale("log")
ax[1, 0].set_yscale("log")
ax[1, 1].set_yscale("log")

ax[0, 0].grid(True)
ax[0, 1].grid(True)
ax[1, 0].grid(True)
ax[1, 1].grid(True)

ax[0, 0].set_xlabel("Communications")
ax[1, 0].set_xlabel("Communications")
ax[0, 1].set_xlabel("Computations")
ax[1, 1].set_xlabel("Computations")

# Generate sequence for communication steps
rng = np.random.default_rng(seed=seed)
comms_seq = np.zeros(iterations)
t_comms = 0
for i in range(iterations):
    if rng.uniform() <= p:
        t_comms += 1
    comms_seq[i] = t_comms
comms_seq = n_c * comms_seq

ax[0, 0].plot(comms_seq, opt_error, label=name)
ax[0, 1].plot(np.arange(iterations), opt_error)
ax[1, 0].plot(comms_seq, con_error)
ax[1, 1].plot(np.arange(iterations), con_error)

lines_labels = [ax1.get_legend_handles_labels() for ax1 in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc="center", bbox_to_anchor=(0.5, -0.01), ncol=6)
fig.tight_layout()
fig.savefig(f"RGTA_experiment_{problem_name}.pdf", bbox_inches="tight")
