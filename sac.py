import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from torch.distributions import Normal

# Soft Actor-Critic:
# Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
# https://arxiv.org/abs/1801.01290
class Actor(nn.Module):
    def __init__(self, state_space, action_space, noise = 1e-6,
            log_std_max = 2, log_std_min = -20):
        super(Actor, self).__init__()
        self.noise = noise
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.l1 = nn.Linear(state_space, 100)
        self.l2 = nn.Linear(100, 500)
        self.l3 = nn.Linear(500, action_space)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean = self.l3(a)
        log_std = self.l3(a)
        log_std = torch.clamp(log_std, min = self.log_std_min,
                max = self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) +
                self.noise)
        entropy = -log_prob.sum(dim = 1, keepdim = True)
        mean = torch.tanh(mean)
        return action, entropy, mean

class Critic(nn.Module):
    def __init__(self, state_space, action_space):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_space + action_space, 100)
        self.l2 = nn.Linear(100, 500)
        self.l3 = nn.Linear(500, 1)

    def forward(self, x):
        q = F.relu(self.l1(x))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q

class DoubleCritic(nn.Module):
    def __init__(self, state_space, action_space):
        super(DoubleCritic, self).__init__()
        self.Q1 = Critic(state_space, action_space)
        self.Q2 = Critic(state_space, action_space)

    def forward(self, state, action):
        x = torch.cat([state, action], 2)
        return self.Q1(x), self.Q2(x)

class SAC(object):
    def __init__(self, state_space, action_space, action_space_shape,
            epsilon = 1e-3, gamma = 0.99):
        self.action_space = action_space
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.policy = Actor(state_space, action_space).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                lr = 3e-4, weight_decay = 1e-1)
        self.critic = DoubleCritic(state_space, action_space).to(self.device)
        self.critic_target = DoubleCritic(state_space, action_space).to(
                self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                lr = 3e-4, weight_decay = 1e-1)
        self.critic_target_optimizer = optim.Adam(
                self.critic_target.parameters(), lr = 3e-4, weight_decay = 1e-1)
        self.critic_target = deepcopy(self.critic)
        for param in self.critic_target.parameters():
            param.requires_grad = False
        # The maximum entropy objective
        self.entropy_target = -torch.prod(torch.Tensor(action_space_shape).to(
            self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad = True,
                device = self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr = 3e-4,
                weight_decay = 1e-1)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        sample = random.random()
        if sample > self.epsilon:
            _, _, action  = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
        return action.max(1)[1].view(1, -1).detach().item()

    def train(self, replay_buffer, prioritized = False):
        if prioritized:
            state, action, next_state, reward, not_done, weight = \
                    replay_buffer.sample()
        else:
            state, action, next_state, reward, not_done = \
                    replay_buffer.sample()
        reward = reward.unsqueeze(1)
        not_done = not_done.unsqueeze(1)
        Q1_current, Q2_current = self.critic(state, action)
        with torch.no_grad():
            next_action, entropy, mean = self.policy.sample(next_state)
            Q1_target, Q2_target = self.critic_target(next_state, next_action)
            Q_target = torch.min(Q1_target, Q2_target) + self.alpha * entropy
        Q_target_hat = reward + (1.0 - not_done) * self.gamma * Q_target
        # J_{Q_1} =
        # E_{(s_t, a_t)}[0.5(Q_1(s_t, a_t) - r(s_t, a_t) - E[V(s{_t+1})])^2]
        Q1_loss = F.mse_loss(Q1_current, Q_target_hat)
        # J_{Q2} =
        # E_{(s_t, a_t)}[0.5(Q_2(s_t, a_t) - r(s_t, a_t) - E[V(s{_t+1})])^2]
        Q2_loss = F.mse_loss(Q2_current, Q_target_hat)
        loss = Q1_loss + Q2_loss
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        action_pi, entropy_pi, _ = self.policy.sample(state)
        Q1_pi, Q2_pi = self.critic(state, action_pi)
        Q_pi = torch.min(Q1_pi, Q2_pi)
        # J_/pi =
        # \E[log_\pif(\epsilon_t;s_t|s_t) - Q(s_t, f(\epsilon_t s_t;s_t)]
        policy_loss = ((self.alpha * entropy_pi) - Q_pi).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph = True)
        self.policy_optimizer.step()
        # Entropy update
        alpha_loss = -(self.log_alpha * (entropy_pi + self.entropy_target
            ).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.slpha = self.log_alpha.exp()
