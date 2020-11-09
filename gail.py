import numpy as np
import torch
from torch import nn
from torch.optim import Adam

# Generative Adversarial Imitation Learning
# https://arxiv.org/abs/1606.03476
class Actor(nn.Module):
    def __init__(self, state_space, action_space, activation, log_std = 0):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.activaten = activation
        self.log_std = log_std
        self.l1 = nn.Linear(self.state_space, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, self.action_space)
        self.log_std = nn.Parameter(torch.ones(1, action_space) * self.log_std)

    def forward(self, states):
        x = self.activation(self.l2(self.l1(states)))
        mean = self.l3(x)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        return mean, log_std, std

class Critic(nn.Module):
    def __init__(self, state_space, activation):
        super().__init__()
        self.state_space = state_space
        self.activation = activation
        self.l1 = nn.Linear(state_space, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, states):
        x = self.activation(self.l2(self.l1(states)))
        value = self.l3(x)
        return value

class Discriminator(nn.Module):
    def __init__(self, state_space, action_space, activation):
        super().__init__()
        self.state_action_space = state_space + action_space
        self.activation = activation
        self.l1 = nn.Linear(self.state_action_space, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, states_actions):
        x = self.l2(self.l1(states_actions))
        x = self.activation(x)
        probability = torch.sigmoid(self.l3(x))
        return probability

class GAIL(object):
    def __init__(self, env, activation, gamma, tau):
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.activation = activation
        self.gamma = gamma
        self.tau = tau
        self.policy = Actor(self.state_space, self.action_space,
                self.activation)
        self.critic = Critic(self.state_space, self.activation)
        self.discriminator = Discriminator(self.state_space, self.action_space,
                self.activation)
        self.policy_optimizer = Adam(self.policy.parameters(), lr = 1e-3)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = 1e-3)
        self.discriminator_optimizer = Adam(self.discriminator.parameters(),
                lr = 1e-3)

    def compute_advantage(self, values, rewards, not_done):
        batch_size = len(rewards)
        advantages = torch.FloatTensor(batch_size)
        value_old = 0.0
        advantage_old = 0
        for i in reversed(range(batch_size)):
            delta = rewards[i] + self.gamma * value_old * not_done[i] - \
                    values.data[i]
            advantages[i] = delta + self.gamma * self.tau * advantage_old * \
                    not_done[i]
            value_old = values.data[i]
            advantage_old = advantages[i]
        returns = values + advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() +
                self.noise)
        return advantages, returns

    def update_parameters(self, batch):
        states = torch.from_numpy(np.stack(batch.state)).float()
        actions = torch.from_numpy(np.stack(batch.action)).long()
        rewards = torch.from_numpy(np.stack(batch.reward)).float()
        not_done = torch.from_numpy(np.stack(batch.not_done)).float()
        with torch.no_grads():
            values = self.critic(states)
            mean, log_std, std = self.policy(states)
            log_probabilities = compute_gaussian_log(actions, mean, log_std,
                    std)
        advantages, returns = self.compute_advantage(values, rewards,
                not_done, self.gamma, self.tau)
