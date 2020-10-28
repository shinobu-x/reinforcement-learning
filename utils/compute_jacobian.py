from itertools import product
import torch
from torch.autograd import grad

def compute_jacobian(f, x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if f.dim() == 0: return grad(f, x)[0]
    jacobian = torch.zeros(f.shape + x.shape).to(device)
    grad_output = torch.zeros(*f.shape).to(device)
    for index in product(*map(range, f.shape)):
        grad_output[index] = 1
        jacobian[index] = grad(f, x, grad_output = grad_output,
                retain_graph = True)[0]
        grad_output[index] = 0
    return jacobian
