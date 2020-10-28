import numpy as np
from casadi import mtimes, norm_2

def compute_maximum_eigenvalues(a, x = None, iters = None):
    r, c = a.shape
    if x is None:
        x = np.eye(c, 1)
        x /= norm_2(x)
    if iters is None:
        iters = 2 * c ** 2
    y = mtimes(a, x)
    for _ in range(iters):
        x = y / norm_2(y)
        y = mtimes(a, x)
    return mtimes(y.T, x)
