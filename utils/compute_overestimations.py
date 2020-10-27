import numpy as np
from scipy import linalg

def compute_overestimations(q, k_fb, l_mu, l_var):
    n_r, n_c = np.shape(k_fb)
    s = np.hstack((np.eye(n_c), k_fb.T))
    b = np.dot(s, s.T)
    qb = np.dot(q, b)
    evals, _ = linalg.eig(qb)
    r_max = np.max(evals)
    upperbound_mu = l_mu * r_max
    upperbound_var = l_var * np.sqrt(r_max)
    return upperbound_mu, upperbound_var
