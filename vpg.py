import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from torch.distributions import Categorical

# Human-level control through deep reinforcement learning
# https://www.nature.com/articles/nature14236
class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_space, 100)
        self.l2 = nn.Linear(100, 500)
        self.l3 = nn.Linear(500, action_space)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        x = F.softmax(x, dim = -1)
        return x

class Critic(nn.Module):
    def __init__(self, state_space):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_space, 100)
        self.l2 = nn.Linear(100, 500)
        self.l3 = nn.Linear(500, 1)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class VPG(object):
    def __init__(self, state_space, action_space, epsilon = 0.09, gamma = 0.99,
            num_episodes = 100, num_timesteps = 10000):
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.state_space = state_space
        self.action_space = action_space
        self.policy = Actor(state_space, action_space).to(self.device)
        self.critic = Critic(state_space).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr = 3e-4,
                weight_decay = 1e-1)
        self.ciritic_optimizer = optim.Adam(self.critic.parameters(),
                lr = 3e-4)
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.num_timesteps = num_timesteps
        self.history = []
        self.log = namedtuple('Log', ('log_probability', 'value', 'reward'))

    def select_action(self, states):
        states = torch.autograd.Variable(torch.from_numpy(states).float()).to(
                self.device)
        probabilities = self.policy(states)
        distribution = Categorical(probs = probabilities)
        action = distribution.sample()
        log_probability = distribution.log_prob
        return action, log_probability

    def train(self, env):
        for episode in range(self.num_episodes):
            episode_reward = 0.0
            states = env.reset()
            for timestep in range(self.num_timesteps):
                action, log_probability = self.select_action(states)
                value = self.critic(torch.autograd.Variable(torch.from_numpy(
                    states).float()).to(self.device))
                next_states, reward, done, _ = env.step(action.item())
                self.history.append(self.log(log_probability, value, reward))
            history = self.log(*zip(*self.history))

import gym
env = gym.make('CartPole-v1')
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
vpg = VPG(state_space, action_space)
vpg.train(env)
