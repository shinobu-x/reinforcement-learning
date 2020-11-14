import torch
import torch.nn.functional as F
from torch import nn
from distributions.categorical import Categorical

# DiCE: The Infinitely Differentiable Monte-Carlo Estimator
# https://arxiv.org/abs/1802.05098
class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_space, 128)
        self.l2 = nn.Linear(128, 128)
        self.dist = Categorical(128, action_space)

    def forward(self, states, action = None, deterministic = False,
            reparam = False):
        x = F.relu(self.l1(states))
        x = F.relu(self.l2(x))
        distribution = self.dist(x)
        entropy = None
        if action is None:
            if deterministic: action = distribution.mode()
            elif reparam: action = distribution.rsample()
            else: action = distribution.sample()
        else:
            entropy = distribution.entropy().mean()
        log_probabilities = distribution.log_prob(action)
        return (action, log_probabilities) if action is None else \
                (entropy, log_probabilities)
