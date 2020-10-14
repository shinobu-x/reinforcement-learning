import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from torch.distributions import Normal
from utils import Filtering
from utils import compute_gaussian_log

# Proximal Policy Optimization Algorithms
# https://arxiv.org/abs/1707.06347
class Actor(nn.Module):
    def __init__(self, state_space, action_space, noise = 1e-6,
            log_std_max = 2, log_std_min = -20):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_space, 100)
        self.l2 = nn.Linear(100, 500)
        self.l3 = nn.Linear(500, action_space)
        self.log_std = nn.Parameter(torch.zeros(1, action_space))
        self.modules = [self.l1, self.l2, self.l3, self.log_std]
        self.modules_old = [None]*len(self.modules)
        self.backup()

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean = self.l3(a)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return mean, log_std, std

    def reference(self, state):
        a = F.relu(self.modules_old[0](state))
        a = F.relu(self.modules_old[1](a))
        mean_old = self.modules_old[2](a)
        log_std_old = self.modules_old[3].expand_as(mean_old)
        std_old = torch.exp(log_std_old)
        return mean_old, log_std_old, std_old

    def backup(self):
        for index in range(len(self.modules)):
            self.modules_old[index] = deepcopy(self.modules[index])

class Critic(nn.Module):
    def __init__(self, state_space):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_space, 100)
        self.l2 = nn.Linear(100, 500)
        self.l3 = nn.Linear(500, 1)

    def forward(self, x):
        q = F.relu(self.l1(x))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q

class PPO(object):
    def __init__(self, state_space, action_space, epsilon = 1e-3, gamma = 0.99,
            tau = 0.9, noise = 1e-4):
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.policy = Actor(state_space, action_space).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                lr = 3e-4, weight_decay = 1e-1)
        self.critic = Critic(state_space).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                lr = 3e-4, weight_decay = 1e-1)
        self.running_state = Filtering(state_space)

    def compute_advantage(self, values, rewards, not_done):
        batch_size = len(rewards)
        value_target = torch.FloatTensor(batch_size)
        advantages = torch.FloatTensor(batch_size)
        value_target_old = 0
        value_old = 0
        advantage_old = 0
        # Compute generalized advantage estimate
        # A^\hat_t = \delta_t + (\gamma\lambda)\delta_{t+1} + ... +
        # (\gamma\lambda)^{T-t+1}\delta_{T-1}
        for i in reversed(range(batch_size)):
            value_target[i] = rewards[i] + self.gamma * value_target_old * \
                    not_done[i]
            delta = rewards[i] + self.gamma * value_old * not_done[i] - \
                    values.data[i]
            advantages[i] = delta + self.gamma * self.tau * advantage_old * \
                    not_done[i]
            value_target_old = value_target[i]
            value_old = values.data[i]
            advantage_old = advantages[i]
        advantages = (advantages - advantages.mean()) / (advantages.std() +
                self.noise)
        return advantages, value_target

    def update_parameters(self, states, actions, rewards, not_done):
        values = self.critic(states)
        advantages, value_target = self.compute_advantage(values, rewards,
                not_done)
        critic_loss = torch.mean(torch.pow(values - torch.autograd.Variable(
            value_target), 2))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        mean, log_std, std = self.policy(states)
        log_probability = compute_gaussian_log(actions, mean, log_std, std)
        with torch.no_grad():
            mean_old, log_std_old, std_old = self.policy.reference(states)
            log_probability_old = compute_gaussian_log(actions, mean_old,
                    log_std_old, std_old)
        self.policy.backup()
        advantages = advantages.unsqueeze(-1)
        self.policy_optimizer.zero_grad()
        ratio = torch.exp(log_probability - log_probability_old)
        # Conservative policy iteration
        surrogate1 = ratio * advantages
        # Penalize changes to the policy that move r_t(\theta) away from 1
        surrogate2 = torch.clamp(ratio, 1.0 - self.epsilon,
                1.0 + self.epsilon) * advantages
        loss = -torch.mean(torch.min(surrogate1, surrogate2))
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        mean, log_std, std = self.policy(state)
        action = torch.normal(mean, std).max(1)[1].view(1, -1).detach().item()
        return action

    def train(self, replay_buffer):
        state, action, advantage, reward, not_done = replay_buffer.sample()
        self.update_parameters(state, action, reward, not_done)
