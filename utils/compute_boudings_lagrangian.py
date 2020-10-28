import numpy as np
from casadi import mtimes
from numpy.linalg import norm
from utils import compute_maximum_eigenvalues

def compute_boudings_lagrangian(q, L, K, k, order = 2):
    s_max = compute_maximum_eigenvalues(q)
    sk_max = compute_maximum_eigenvalues(mtimes(K, mtimes(q, K.T)))
    if order == 2:
        l_max = s_max ** 2 + sk_max ** 2
        lower_bound = -L * l_max * 0.5
        upper_bound = L * l_max * 0.5
    else:
        l_max = s_max + sk_max
        lower_bound = -L * l_max
        upper_bound = L * l_max
    return lower_bound, upper_bound
