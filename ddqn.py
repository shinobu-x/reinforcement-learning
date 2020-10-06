import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Dueling Network Architectures for Deep Reinforcement Learning
# https://arxiv.org/abs/1511.06581
class Agent(nn.Module):
    def __init__(self, state_space, action_space):
        super(Agent, self).__init__()
        self.l1 = nn.Linear(state_space, 100)
        self.v1 = nn.Linear(100, 500)
        self.a1 = nn.Linear(100, 500)
        self.v2 = nn.Linear(500, 1)
        self.a2 = nn.Linear(500, action_space)

    def forward(self, state):
        a = F.relu(self.l1(state))
        v = F.relu(self.v1(a))
        a = F.relu(self.a1(a))
        v = self.v2(v)
        a = self.a2(a)
        q = v + a - torch.mean(a, dim = 1, keepdim = True)
        return q

class DDQN(object):
    def __init__(self, state_space, action_space, gamma = 0.99):
        self.action_space = action_space
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.policy = Agent(state_space, action_space).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                weight_decay = 1e-1)
        self.epsilon = 0.09
        self.gamma = gamma

    def select_action(self, state):
        sample = random.random()
        if sample > self.epsilon:
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
            return self.policy(state).max(1)[1].view(1, 1).detach().item()
        else:
            return torch.tensor([[random.randrange(self.action_space)]],
                    dtype = torch.long).item()

    def train(self, replay_buffer, prioritized = False):
        if prioritized:
            state, action, next_state, reward, not_done, weight = \
                    replay_buffer.sample()
        else:
            state, action, next_state, reward, not_done = replay_buffer.sample()
        Q_current = self.policy(state)
        Q_current = Q_current.gather(0, action).squeeze(1)
        Q_target = self.policy(next_state)
        Q_target = Q_target.max(1)[0].detach()
        Q_target = reward + self.gamma * Q_target
        self.policy_optimizer.zero_grad()
        if prioritized:
            loss = torch.abs(Q_current.float() - Q_target.float()) * \
                    torch.from_numpy(weight).to(self.device)
            loss = loss.mean()
        else:
            loss = F.mse_loss(Q_current.float(), Q_target.float())
        loss.backward()
        self.policy_optimizer.step()
