import numpy as np
import torch
from scipy import signal

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

def flatten(list):
    return [item for sub_list in list for item in sub_list]

def compute_discounted_reward(rewards, gamma):
    return signal.lfilter([1], [1, -gamma], rewards[::-1], axis = 0)[::-1]

def explained_variance_1d(y_predict, y_target):
    assert y_predict.ndim == 1 and y_target.ndim == 1
    variance = np.var(y_target)
    return np.nan if variance == 0 \
            else 1 - np.var(y_target - y_predict) / variance

def compute_normalized_reward(reward):
    mean = reward.mean()
    std = reward.std()
    return (reward - mean) / (std + 1e-8)
