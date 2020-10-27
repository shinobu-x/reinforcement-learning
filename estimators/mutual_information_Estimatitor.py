import math
import torch
import torch_lightning as pl
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F
from torch.optim import Adam
from torch_lightning import Trainer
from ..utils.misc import get_batch

# MINE: Mutual Information Neural Estimation
# https://arxiv.org/abs/1801.04062
def compute_exponential_moving_average(mu, alpha, running_mean):
    return alpha * mu + (1.0 - alpha) * running_mean

def exponential_moving_average_loss(x, running_mean, alpha):
    exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])
            ).detach()
    if runnin_mean == 0:
        running_mean = exp
    else:
        running_mean = compute_exponential_moving_average(exp, alpha,
                running_mean.item())
    log = ExponentialMovingAverageLoss.apply(x, running_mean)
    return log, running_mean

class ExponentialMovingAverageLoss(Function):
    @staticmethod
    def forward(context, x, running_exponential_moving_average):
        contex.save_for_backward(x, running_exponential_moving_average)
        x_logsumexp = x.exp().mean().log()
        return x_logsumexp

    @staticmethod
    def backward(context, grad_output):
        x, running_mean = context.save_tensors
        grad = grad_output * x.exp().detach() / (running_mean + epsilon) / \
                x.shape[0]
        return grad, None

class BaseModel(nn.Module):
    def __init__(self, x_shape, z_shape):
        self.layers = nn.Sequential(
                nn.Linear(x_shape + z_shape, 400),
                nn.ReLU(),
                nn.Linear(400, 400),
                nn.ReLU(),
                nn.Linear(400, 400),
                nn.ReLU(),
                nn.Linear(400, 1))

    def forward(self, x, z):
        return self.layer(x, z)

class MutualIformationEstimator(nn.Module):
    def __init__(self, x_shape, z_shape, loss = None, alpha = 1e-2,
            method = None):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.model = BaseModel(x_shape, z_shape)
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')

    def forward(self, x, z, z_marginal = None):
        if z_marginal is None:
            z_marginal = z[torch.randperm(x.shape[0])]
        first_term = mean = self.model(x, z).mean()
        mean_marginal = self.model(x, z_marginal)
        second_term, self.running_mean = exponential_moving_average_loss(
                z_marginal, self.running_mean, self.alpha)
        # second_term = torch.exp(z_marginal - 1).mean() # F-divergence
        # V(\theta) <- 1/b * \sum^b_{i=1} T_\theta(x^i, z^i) -
        # log(1/b * \sum^b_{i=1} * e^{T_\theta(x^i, z^i)})
        return first_term - second_term

    def compute_mutual_information(self, x, z, z_marginal = None):
        with torch.no_grad():
            return -self.forward(x, z, z_marginal)

    def train(self, x, z, num_iters, batch_size, optimizer):
        if optimizer is None:
            optimizer = Adam(self.parameters(), lr = 1e-4)
        for i in range(1, num_iters + 1):
            mu = 0
            for x_i, z_i in get_batch(x, z, batch_size):
                optimizer.zero_grad()
                loss = self.forward(x_i, z_i)
                loss.backward()
                optimizer.step()
                mu -= loss.item()
        return compute_mutual_information(x, z)
