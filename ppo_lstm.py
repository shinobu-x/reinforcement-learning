import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from utils import Filtering
from utils import compute_gaussian_log

# Proximal Policy Optimization Algorithms
# https://arxiv.org/abs/1707.06347
class Actor(nn.Module):
    def __init__(self, state_space, action_space, noise = 1e-6,
            log_std_max = 2, log_std_min = -20):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_space, 100)
        self.l2 = nn.Linear(100, 512)
        self.l3 = nn.Linear(256, action_space)
        self.lstm = nn.LSTMCell(512, 256)
        self.device = torch.device('cuda' if torch.cuda.is_available() \
                else 'cpu')
        self.hx = torch.autograd.Variable(torch.zeros(1, 256)).to(self.device)
        self.cx = torch.autograd.Variable(torch.zeros(1, 256)).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(1, action_space))
        self.modules = [self.l1, self.l2, self.lstm, self.l3, self.log_std]
        self.modules_old = [None]*len(self.modules)
        self.backup()

    def forward(self, state, hx, cx):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        hx, cx = self.lstm(a.view(a.size(0), -1)[:1], (hx, cx))
        mean = self.l3(hx)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return mean, log_std, std

    def reference(self, state, hx, cx):
        a = F.relu(self.modules_old[0](state))
        a = F.relu(self.modules_old[1](a))
        hx, cx = self.modules_old[2](a.view(a.size(0), -1)[:1], (hx, cx))
        mean_old = self.modules_old[3](hx)
        log_std_old = self.modules_old[4].expand_as(mean_old)
        std_old = torch.exp(log_std_old)
        return mean_old, log_std_old, std_old

    def backup(self):
        for index in range(len(self.modules)):
            self.modules_old[index] = deepcopy(self.modules[index])

class Critic(nn.Module):
    def __init__(self, state_space, batch_size):
        super(Critic, self).__init__()
        self.batch_size = batch_size
        self.input_size = 512
        self.hidden_size = 256
        self.l1 = nn.Linear(state_space, 100)
        self.l2 = nn.Linear(100, self.input_size)
        self.l3 = nn.Linear(self.hidden_size // self.batch_size, 1)
        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() \
                else 'cpu')
        self.hx = torch.autograd.Variable(torch.zeros(1, self.hidden_size)).to(
                self.device)
        self.cx = torch.autograd.Variable(torch.zeros(1, self.hidden_size)).to(
                self.device)

    def forward(self, x, hx, cx):
        q = F.relu(self.l1(x))
        q = F.relu(self.l2(q))
        hx, cx = self.lstm(q.view(q.size(0), -1)[:1], (hx, cx))
        q = self.l3(hx.view(self.batch_size, 1, -1))
        return q

class PPOLSTM(object):
    def __init__(self, state_space, action_space, batch_size, epsilon = 1e-3,
            gamma = 0.99, tau = 0.9, noise = 1e-4):
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.policy = Actor(state_space, action_space).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                lr = 3e-4, weight_decay = 1e-1)
        self.critic = Critic(state_space, self.batch_size).to(self.device)
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
        self.policy.hx = torch.autograd.Variable(self.policy.hx.data).to(
                self.device)
        self.policy.cx = torch.autograd.Variable(self.policy.cx.data).to(
                self.device)
        self.critic.hx = torch.autograd.Variable(self.critic.hx.data).to(
                self.device)
        self.critic.cx = torch.autograd.Variable(self.critic.cx.data).to(
                self.device)
        values = self.critic(states, self.critic.hx, self.critic.cx)
        advantages, value_target = self.compute_advantage(values, rewards,
                not_done)
        critic_loss = torch.mean(torch.pow(values - torch.autograd.Variable(
            value_target), 2))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        mean, log_std, std = self.policy(states, self.policy.hx, self.policy.cx)
        log_probability = compute_gaussian_log(actions, mean, log_std, std)
        with torch.no_grad():
            mean_old, log_std_old, std_old = self.policy.reference(states,
                    self.policy.hx, self.policy.cx)
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
