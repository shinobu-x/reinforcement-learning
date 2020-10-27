import numpy as np
from numpy.linalg import norm

def compute_bounding_box(q, L, K, k, order = 2):
    s_max = norm(q, ord = 2)
    sk_max = norm(np.dot(K, np.dot(q, K.T)), ord = 2)
    if order == 2:
        l_max = s_max ** 2 + sk_max ** 2
        lower_bound = -L * l_max * (1.0 / order)
        upper_bound = L * l_max * (1.0 / order)
    if order == 1:
        l_max = s_max + sk_max
        lower_bound = -L * l_max
        upper_bound = L * l_max
    return lower_bound, upper_bound
