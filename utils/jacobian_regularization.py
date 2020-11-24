import torch
from torch import autograd
from torch import nn

# Robust Learning with Jacobian Regularization
# https://arxiv.org/abs/1908.02729
class JacobianRegularization(nn.Module):
    def __init__(self, num_projections = 1):
        self.num_projections = num_projections
        super(JacobianRegularization, self).__init__()

    def forward(self, x, z):
        b, c = z.shape
        if self.num_projections == -1: self.num_projections = c
        J_F = 0.0
        for i in range(self.num_projections):
            if self.num_projections == c:
                v = torch.zeros(b, c)
                v[:, i] = 1
            else: v = self.compute_normalized_vector(b, c)
            # J_v = \partial (z_{flat} * v_{flat} / \partial x^{alpha}
            j = self.compute_jacobian_vector(x, z, v, create_graph = True)
            # J_F += C * ||J_v||^2 / (n_{proj}|\mathcal{B}|
            J_F += c * torch.norm(j) ** 2 / (self.num_projections * b)
        return (1 / 2) * J_F

    def compute_normalized_vector(self, batch_size, num_channels):
        if num_channels == 1: return torch.ones(batch_size)
        v = torch.randn(batch_size, num_channels)
        t = torch.zeros(batch_size, num_channels)
        norm = torch.norm(v, 2, 1, True)
        v = torch.addcdiv(t, 1.0, v, norm)
        return v

    def compute_jacobian_vector(self, x, z, v, create_graph = False):
        z_flat = z.reshape(-1)
        v_flat = v.reshape(-1)
        grad_x = torch.autograd.grad(z_flat, x, v_flat, retain_graph = True,
                create_graph = create_graph)
        return grad_x
