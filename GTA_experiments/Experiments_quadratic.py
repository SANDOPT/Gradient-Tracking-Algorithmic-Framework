import numpy as np
from matplotlib import pyplot as plt

import sys

sys.path.append("../")
import Graphs
import problems
import Algorithms


n = 16  # number of nodes
n_c = 5  # number of communication steps
n_g = 5  # number of computation steps
iterations = int(1e4)  # number of iterations

"""Decentralized Graphs"""
# W = Graphs.W_full(n)
W = Graphs.W_star(n, 0.05)
# W = Graphs.W_cyclic(n, 0.05)
# W = Graphs.W_line(n, 0.05)
# W = Graphs.W_full_almost_2(n)
# W = Graphs.W_full_almost_1(n)

"""Quadratic problem definition"""
problem_name = "Quadratic"
d = 10  # dimension of problem
xi = 2  # problem conditioning
A, b, A_list, b_list = problems.Quadratic_problem_generator(d, xi, n, seed=10)
x_init = np.zeros([n, d])  # initial point
x_opt = -np.linalg.solve(A, b)  # cenrtalized problem solution
grad_full = lambda x: problems.Quadratic_decentralized_gradient(
    A_list=A_list, b_list=b_list, x=x
)

"""Uncomment to run GTA algorithms"""
# The step size (alpha) must be changed according to the problem and graph selected
"""Uncomment for GTA1"""
# opt_error, con_error = Algorithms.GTA1(
#     x_0=x_init,
#     W=W,
#     alpha=1e-3,
#     iterations=iterations,
#     grad_full=grad_full,
#     n_g=n_g,
#     n_c=n_c,
#     x_opt=x_opt,
# )
# name = f"GTA-1({n_c}, {n_g})"
"""Uncomment for GTA2"""
# opt_error, con_error = Algorithms.GTA2(
#     x_0=x_init,
#     W=W,
#     alpha=1e-3,
#     iterations=iterations,
#     grad_full=grad_full,
#     n_g=n_g,
#     n_c=n_c,
#     x_opt=x_opt,
# )
# name = f"GTA-2({n_c}, {n_g})"
"""Uncomment for GTA3"""
opt_error, con_error = Algorithms.GTA3(
    x_0=x_init,
    W=W,
    alpha=1e-3,
    iterations=iterations,
    grad_full=grad_full,
    n_g=n_g,
    n_c=n_c,
    x_opt=x_opt,
)
name = f"GTA-3({n_c}, {n_g})"

"""Plotting code"""
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
fig.suptitle(problem_name)
ax[0, 0].set_ylabel("Optimization Error")
ax[1, 0].set_ylabel("Consensus Error")

ax[0, 0].set_yscale("log")
ax[0, 1].set_yscale("log")
ax[0, 2].set_yscale("log")
ax[1, 0].set_yscale("log")
ax[1, 1].set_yscale("log")
ax[1, 2].set_yscale("log")

ax[0, 0].grid(True)
ax[0, 1].grid(True)
ax[0, 2].grid(True)
ax[1, 0].grid(True)
ax[1, 1].grid(True)
ax[1, 2].grid(True)

ax[0, 0].set_xlabel("Iterations")
ax[1, 0].set_xlabel("Iterations")
ax[0, 1].set_xlabel("Communications")
ax[1, 1].set_xlabel("Communications")
ax[0, 2].set_xlabel("Computations")
ax[1, 2].set_xlabel("Computations")

ax[0, 0].plot(np.arange(iterations), opt_error, label=name)
ax[0, 1].plot(n_c * np.arange(iterations), opt_error)
ax[0, 2].plot(n_g * np.arange(iterations), opt_error)
ax[1, 0].plot(np.arange(iterations), con_error)
ax[1, 1].plot(n_c * np.arange(iterations), con_error)
ax[1, 2].plot(n_g * np.arange(iterations), con_error)

lines_labels = [ax1.get_legend_handles_labels() for ax1 in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc="center", bbox_to_anchor=(0.5, -0.01), ncol=6)
fig.tight_layout()
fig.savefig(f"GTA_experiment_{problem_name}.pdf", bbox_inches="tight")
