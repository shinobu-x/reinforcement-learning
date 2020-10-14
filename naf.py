import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Continuous Deep Q-Learning with Model-based Acceleration
# https://arxiv.org/abs/1603.00748
class Agent(nn.Module):
    def __init__(self, state_space, action_space):
        super(Agent, self).__init__()
        self.action_space = action_space
        self.bn = nn.BatchNorm1d(state_space)
        self.l1 = nn.Linear(state_space, 100)
        self.l2 = nn.Linear(100, 500)
        self.l3 = nn.Linear(500, 1)
        self.l4 = nn.Linear(500, action_space)
        self.l5 = nn.Linear(500, action_space ** 2)
        self.triangular_mask = torch.autograd.Variable(torch.tril(torch.ones(
            action_space, action_space), diagonal = -1).unsqueeze(0))
        self.diagonal_mask = torch.autograd.Variable(torch.diag(torch.ones(
            action_space, action_space))).unsqueeze(0)

    def forward(self, state, action = None):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        V = self.l3(x)
        mu = F.relu(self.l4(x))
        # Q(x, mu|\theta^Q) = A(x, mu|\theta^A) + V(x|\theta^V)
        Q = None
        if action is not None:
            # P(x|\theta^P) = L(x|\theta^P) * L(x|\theta^P)^T
            L = self.l5(x).view(-1, self.action_space, self.action_space)
            L = L * self.triangular_mask.expand_as(L) + torch.exp(L) * \
                    self.diagonal_mask.expand_as(L)
            P = torch.bmm(L, L.transpose(2, 1))
            # A(x, mu|\theta^A) = 0.5 * (u - mu(x|\theta^mu))^T *
            # P(x|\theta^P) * (u - mu(x|\theta^mu))
            mu = (action - mu)
            a = torch.bmm(mu.transpose(2, 1), P)
            A = -0.5 * torch.bmm(a, mu)[:, :, 0]
            Q = A + V
        return mu, Q, V

class NAF(object):
    def __init__(self, state_space, action_space, gamma = 0.99, tau = 1e-3,
            noise = 1e-5):
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.policy = Agent(state_space, action_space).to(self.device)
        self.policy_target = Agent(state_space, action_space).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = 1e-4)
        for param, param_target in zip(self.policy.parameters(),
                self.policy_target.parameters()):
            param_target.data.copy_(param.data)

    def update_parameters(self, states, actions, next_states, rewards):
        _, value, _ = self.policy(states, actions)
        _, _, value_target = self.policy_target(next_states)
        value_target = rewards + (value_target * self.gamma)
        loss = F.mse_loss(value, value_target)
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        for param, param_target in zip(self.policy.parameters(),
                self.policy_target.parameters()):
            # Update the target network
            param_target.data.copy_(self.tau * param.data +
                (1 - self.tau) * param_target.data)

    def select_action(self, state):
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            mu, _, _ = self.policy(state)
            mu.data += self.noise
            return mu.data.clamp(-1, 1)

    def train(self, replay_buffer):
        state, action, next_state, reward, not_done = replay_buffer.sample()
        self.update_parameters(state, action, next_state, reward)
