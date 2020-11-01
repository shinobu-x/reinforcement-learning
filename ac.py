import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import nn
from torch.optim import Adam

class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_space, 128)
        self.l2 = nn.Linear(128, action_space)

    def forward(self, state):
        x = F.relu(self.l1(state))
        probabilities = F.softmax(self.l2(x), -1)
        return probabilities

class Critic(nn.Module):
    def __init__(self, state_space):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_space, 128)
        self.l2 = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = self.l2(x)
        return x

History = namedtuple('History', ['log_prob', 'value'])
class AC(object):
    def __init__(self, state_space, action_space, gamma = 0.99):
        self.state_space = state_space
        self.action_space = action_space
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.policy = Actor(self.state_space, self.action_space).to(
                self.device)
        self.critic = Critic(self.state_space).to(self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr = 3e-2)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = 3e-2)
        self.gamma = gamma
        self.epsilon = np.finfo(np.float32).eps.item()
        self.history = []
        self.rewards = []

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        probs = self.policy(state)
        value = self.critic(state)
        m = Categorical(probs)
        action = m.sample()
        self.history.append(History(m.log_prob(action), value))
        return action.item()

    def finish_episode(self):
        R = 0
        history = agent.history
        policy_losses = []
        value_losses = []
        returns = []
        for r in agent.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() +
                self.epsilon)
        for (log_prob, value), R in zip(history, returns):
             advantage = R - value.item()
             policy_losses.append(-log_prob * advantage)
             value_losses.append(F.smooth_l1_loss(value,
                 torch.tensor([R]).to(self.device)))
        agent.policy_optimizer.zero_grad()
        agent.critic_optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum()
        loss += torch.stack(value_losses).sum()
        loss.backward()
        agent.policy_optimizer.step()
        agent.critic_optimizer.step()
        del agent.rewards[:]
        del agent.history[:]

env = gym.make('CartPole-v0')
state_space = env.observation_space.shape[0]
action_space = env.action_space.n
agent = AC(state_space, action_space)
num_episodes = 100000
num_timesteps = 10000
total_discounted_reward = 10
gamma = 0.05
capacity = 10000
for episode in range(1, num_episodes):
    state = env.reset()
    episode_reward = 0
    for t in range(1, num_timesteps):
        action = agent.select_action(state)
        state, reward, done, _ = env.step(action)
        if False:
            env.render()
        agent.rewards.append(reward)
        episode_reward += reward
        if done:
            break
    total_discounted_reward = gamma * episode_reward + \
            (1 - gamma) * total_discounted_reward
    agent.finish_episode()
    if episode % 10 == 0:
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.
                format(episode, episode_reward, total_discounted_reward))
    if total_discounted_reward > env.spec.reward_threshold:
        print("Average Reward: {}\nThe last episode runs: {} time steps".
                format(total_discounted_reward, t))
        break
