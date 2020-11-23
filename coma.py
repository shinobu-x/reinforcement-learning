import torch
import torch.nn.functional as F
from collections import namedtuple
from torch import nn, optim
from distributions.categorical import Categorical
from replay_buffer.distribution import ReplayBuffer

# Counterfactual Multi-Agent Policy Gradients
# https://arxiv.org/abs/1705.08926
class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_space, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, action_space)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return F.softmax(self.l3(x), dim = -1)

class Critic(nn.Module):
    def __init__(self, state_space, action_space, num_agents):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_space * num_agents ** 2 + 1, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, action_space)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.l3(x)

class COMA(object):
    def __init__(self, state_space, action_space, num_agents, gamma):
        self.state_space = state_space
        self.action_space = action_space
        self.num_actions = num_actions
        self.gamma = gamma
        self.policy = [Actor(state_space, action_space)
                for _ in range(num_agents)]
        self.critic = Critic(state_space, action_space, num_agents)
        self.critic_target = Critic(state_space, action_space, num_agents)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.policy_optimizer = [optim.Adam(self.policy[i].parameters(),
            lr = 1e-3) for i in range(num_agents)]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = 1e-3)
        self.count = 0

    def select_action(self, state):
        state = torch.tensor(state)
        actions = []
        for i in range(self.num_agents):
            distribution = self.policy[i](state[i])
            action = Categorical(distribution).sample()
        return action

    def train(self, replay_buffer):
        actions, states, distribution, reward, not_done = replay_buffer.sample()
        for i in range(self.num_agents):
            batch_size = len(states)
            ids = (torch.ones(batch_size) * i).view(-1, 1)
            states = torch.cat(states).view(batch_size,
                    self.state_space * self.num_agents)
            states = torch.cat([states.type(torch.float32),
                actions.type(torch.float32)], dim = -1)
            states = torch.cat([ids, states], dim = -1)
            Q_target = self.critic_target(states)
            action = actions.type(torch.long)[:, i].reshape(-1, 1)
            baseline = torch.sum(distribution[i] * Q_target, dim = 1).detach()
            Q_target = torch.gather(distribution[i], dim = 1,
                    index = action).squeeze()
            advantage = Q_target - baseline
            log_probability = torch.log(torch.gather(distribution[i], dim = 1,
                index = action).squeeze())
            policy_loss = torch.mean(advantage * log_probability)
            self.policy_optimizer[i].zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy[i].parameters(), 5)
            self.policy_optimizer.step()
            Q_current = self.critic(states)
            action = actions.type(torch.long)[:, i].reshape(-1, 1)
            Q_current = torch.gather(Q, dim = 1, index = action).squeeze()
            R = torch.zeros(len(reward[:, i]))
            for j in range(len(reward[:, i])):
                if done[i][j]: R[j] = reward[:, i][j]
                else: R[j] = reward[:, i][j] * self.gamma * Q_target[j + 1]
            critic_loss = torch.mean((R - Q) ** 2)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
            self.critic_optimizer.step()
        if self.count == 10000:
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.count = 0
        else: self.count += 1
