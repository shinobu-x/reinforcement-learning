import numpy as np
import torch
from scipy import signal
from torch import nn
from utils.compute_jacobian import compute_jacobian
from utils.compute_maximum_eigenvalues import compute_maximum_eigenvalues
from utils.compute_overestimations import compute_overestimations
from utils.compute_orthogonal_regularization import \
        compute_orthogonal_regularization

class RunningState(object):
    def __init__(self, shape):
        self._n = 0
        self._m = np.zeros(shape)
        self._s = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self.mean.shape
        self._n += 1
        if self._n == 1:
            self.mean[...] = x
        else:
            m_old = self._m.copy()
            self._m[...] = m_old + (x - m_old) / self._n
            self._s[...] = self._s + (x - m_old) * (x - self._m)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._m

    @property
    def var(self):
        return self._s / (self._n - 1) if self._n > 1 else \
                np.square(self._m)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._m.shape

class Filtering:
    def __init__(self, shape, de_mean = True, de_std = True, epsilon = 10.0):
        self.de_mean = de_mean
        self.de_std = de_std
        self.epsilon = epsilon
        self.running_state = RunningState(shape)

    def __call__(self, x, update = True):
        if update:
            self.running_state.push(x)
        if self.de_mean:
            x = x - self.running_state.mean
        if self.de_std:
            x = x / (self.running_state.std + 1e-8)
        if self.epsilon:
            x = np.clip(x, -self.epsilon, self.epsilon)
        return x

def p2v(parameters):
    vector = []
    for param in parameters:
        vector.append(param.view(-1))
    return torch.cat(vector)

def v2p(vector, parameters):
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        param.data = vector[pointer:pointer + num_param].view_as(param).data
        pointer += num_param

def init_weight(m):
    classname = m.__class__.__name__
    weight_shape = list(m.weight.data.size())
    m.bias.data.fill_(0)
    if classname.find('Conv') != -1:
        l = np.prod(weight_shape[1: 4])
        r = np.prod(weight_shape[2: 4]) * weight_shape[0]
    elif classname.find('Linear') != -1:
        l = np.prod(weight_shape[1])
        r = np.prod(weight_shape[0])
    bound = np.sqrt(6.0 / (l + r))
    m.weight.data.uniform_(-bound, bound)

def columnwise_init(weights, std = 1.0):
    x = torch.randn(weights.size())
    return std / torch.sqrt((x ** 2).sum(1, keepdim = True))

def flatten(list):
    return [item for sub_list in list for item in sub_list]

def compute_discounted_reward(rewards, gamma):
    return signal.lfilter([1], [1, -gamma], rewards[::-1], axis = 0)[::-1]

def explained_variance_1d(y_predict, y_target):
    assert y_predict.ndim == 1 and y_target.ndim == 1
    variance = np.var(y_target)
    return np.nan if variance == 0 \
            else 1 - np.var(y_target - y_predict) / variance

def compute_gaussian_log(x, mean, log_std, std):
    var = std.pow(2)
    pi = torch.autograd.Variable(torch.DoubleTensor([[3.1415926]]))
    return -(x - mean).pow(2) / (2 * var) - 0.5 * torch.log(2 * pi) - log_std

def compute_kl_divergence(mean_p, std_p, mean_q, std_q, noise = 1e-8):
    numerator = torch.pow((mean_p - mean_q), 2.0) + torch.pow(std_p, 2.0) - \
            torch.pow(std_q, 2.0)
    denominator = 2.0 * torch.pow(std_q, 2.0) + noise
    return torch.sum(numerator / denominator + torch.log(std_q) -
            torch.log(p_std))

def compute_entropy(std):
    return torch.sum(std * 0.5 * torch.log(2.0 * np.pi * np.e))

def compute_normal_entropy(std):
    var = std.pow(2)
    return (0.5 ** 2 * torch.log(2 * var * np.pi)).sum(1)
