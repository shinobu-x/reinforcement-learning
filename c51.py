import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy

# A Distributional Perspective on Reinforcement Learning
# https://arxiv.org/abs/1707.06887
class Agent(nn.Module):
    def __init__(self, state_space, action_space, atoms):
        super(Agent, self).__init__()
        self.action_space = action_space
        self.atoms = atoms
        self.l1 = nn.Linear(state_space, 100)
        self.l2 = nn.Linear(100, 500)
        self.l3 = nn.Linear(500, action_space * atoms)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = x.view(-1, self.action_space, self.atoms)
        x = F.log_softmax(x, 2).exp()
        return x

class C51(object):
    def __init__(self, state_space, action_space, epsilon = 0.09, gamma = 0.99,
            atoms = 51, vmax = 10, vmin = -10):
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.atoms = atoms
        self.vmax = vmax
        self.vmin = vmin
        self.support = torch.linspace(vmin, vmax, atoms).view(1, 1, atoms
                ).to(self.device)
        self.delta = (vmax - vmin) / (atoms - 1)
        self.policy = Agent(state_space, action_space, atoms).to(self.device)
        self.policy_target = Agent(state_space, action_space, atoms
                ).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = 3e-4,
                weight_decay = 1e-1)
        self.policy_target = deepcopy(self.policy)

    def select_action(self, state):
        sample = random.random()
        if sample > self.epsilon:
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            x = self.policy(state)
            x = (x * self.support).sum(2).max(1)[1].view(1, 1)
            return x.detach().item()
        else:
            return torch.tensor([[random.randrange(self.action_space)]],
                    dtype = torch.long).item()

    def train(self, replay_buffer):
        state, action, next_state, reward, not_done = replay_buffer.sample()
        batch_size = state.shape[0]
        action = action.squeeze(2).view(-1, 1, 1).expand(-1, 1, self.atoms)
        reward = reward.view(-1, 1, 1)
        Q = self.policy(state)
        distribution_current = Q.view(-1, 1, self.atoms).gather(0,
                action).squeeze()
        distribution_current = distribution_current[64::]
        non_final = tuple(map(lambda s: s is not None, next_state))
        mask = torch.tensor(non_final, device = self.device,
                dtype = torch.bool)
        with torch.no_grad():
            distribution_target = torch.zeros((batch_size, 1, self.atoms),
                    device = self.device, dtype = torch.float)
            distribution_target += 1.0 / self.atoms
            Q_target = self.policy_target(next_state)
            # Q(x_{t+1}, a) := \Sigma_i z_i * p_i(x_{t+1}, a)
            Q_target = (Q_target * self.support).sum(2)
            action_target = Q_target.max(1)[1]
            action_target = action_target.view(batch_size, 1, 1).expand(
                    -1, -1, self.atoms)
            distribution_target[mask] = self.policy_target(next_state).gather(
                    1, action_target)
            distribution_target = distribution_target.squeeze()
            # T^{hat}_{z_j} <- [r_t + r_t * z_j]^{V_{MAX}}_{V_{MIN}}
            z = reward.view(-1, 1) + (self.gamma ** 3) * self.support.view(
                    1, -1) * mask.to(torch.float).view(-1, 1)
            z = z.clamp(self.vmin, self.vmax)
            # b <- (T^{hat}_{z_j} - V_{MIN} / \delta_z
            b = (z - self.vmin) / self.delta
            # l <- \lfoor b_j \rfloor
            l = b.floor().to(torch.int64)
            # u <- \lceil b_j \rceil
            u = b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1
            offset = torch.linspace(0, (batch_size - 1) * self.atoms,
                    batch_size).unsqueeze(1).expand(batch_size, self.atoms).to(
                            action)
            distribution_target = state.new_zeros(batch_size, self.atoms)
            # m_l <- m_l + p_j(x_{t+1}, a^\ast)(u - b_j)
            distribution_target.view(-1).index_add_(0, (l + offset).view(-1),
                    (distribution_target * (u.float() - b)).view(-1))
            # m_u <- m_u + p_j(x_{t+1}, a^\ast)(b_j - l)
            distribution_target.view(-1).index_add_(0, (u + offset).view(-1),
                    (distribution_target * (b - l.float())).view(-1))
        # -\Sigma_i m_i * log p_i(x_t, a_t)
        loss = -(distribution_target * distribution_current.log()).sum(-1)
        loss = loss.mean()
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
