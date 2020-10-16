import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

# Reinforcement Learning with Deep Energy-Based Policies
# https://arxiv.org/abs/1702.08165
class Agent(nn.Module):
    def __init__(self, state_space, action_space, alpha = 2.0):
        super(Agent, self).__init__()
        self.alpha = alpha
        self.l1 = nn.Linear(state_space, 100)
        self.l2 = nn.Linear(100, 500)
        self.l3 = nn.Linear(500, action_space)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)
        return a

    def compute_value(self, q):
        # V = \alpha * logE[exp(1/alpha * Q^\theta(s_t, a^\prime)]
        return self.alpha * torch.log(torch.sum(torch.exp(q / self.alpha),
            dim = 1, keepdim = True))

class SQL(object):
    def __init__(self, state_space, action_space, gamma = 0.99, alpha = 2.0):
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.policy = Agent(state_space, action_space,
                alpha = alpha).to(self.device)
        self.policy_target = Agent(state_space, action_space,
                alpha = alpha).to(self.device)
        self.policy_target = deepcopy(self.policy)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = 3e-4,
                weight_decay = 1e-1)
        self.alpha = alpha
        self.gamma = gamma

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            Q = self.policy(state)
            V = self.policy.compute_value(Q)
            # D_KL(\pi^\phi( |s_t) | exp(1/alpha * (Q^\theta(s_t, ) -
            # V^\theta)))
            distribution = torch.exp((Q - V) / self.alpha)
            distribution = distribution / torch.sum(distribution)
            a = Categorical(distribution)
            a = a.sample()
        return a.item()

    def train(self, replay_buffer):
        state, action, next_state, reward, not_done = replay_buffer.sample()
        reward = reward.unsqueeze(1)
        not_done = not_done.unsqueeze(1)
        value_current = self.policy(state).gather(0, action.long())
        with torch.no_grad():
            Q_next = self.policy_target(next_state)
            V_next = self.policy_target.compute_value(Q_next)
            value_target = reward + self.gamma * V_next * not_done
        loss = F.mse_loss(value_current, value_target)
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
