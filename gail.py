import numpy as np
import torch
import torch.nn.functional as F
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

    def forward(self, state):
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
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.activation = activation
        self.gamma = gamma
        self.tau = tau
        self.clip_threshold = 1e-3
        self.l2_reguralization = 1e-1
        self.policy = Actor(self.state_space, self.action_space,
                self.activation)
        self.critic = Critic(self.state_space, self.activation).to(
                self.device)
        self.discriminator = Discriminator(self.state_space, self.action_space,
                self.activation).to(self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr = 1e-3)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = 1e-3)
        self.discriminator_optimizer = Adam(self.discriminator.parameters(),
                lr = 1e-3)
        self.discriminator_criterion = nn.BCELoss()

    def select_action(self, state):
        mean, _, std = self.policy.forward(state)
        action = torch.nomal(mean, std)
        return action

    def compute_advantage(self, values, rewards, not_done):
        batch_size = len(rewards)
        advantages = torch.FloatTensor(batch_size).to(self.device)
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
        states = torch.from_numpy(np.stack(batch.state)).float().to(self.device)
        actions = torch.from_numpy(np.stack(batch.action)).long().to(
                self.device)
        rewards = torch.from_numpy(np.stack(batch.reward)).float().to(
                self.device)
        not_done = torch.from_numpy(np.stack(batch.not_done)).float().to(
                self.device)
        with torch.no_grads():
            values = self.critic(states)
            mean, log_std, std = self.policy(states)
            log_probabilities_target = compute_gaussian_log(
                    actions, mean, log_std, std)
        advantages, returns = self.compute_advantage(values, rewards,
                not_done, self.gamma, self.tau)
        discriminator_optimizer.zero_grad()
        generator_value = discriminator(torch.cat([states, actions], 1))
        expert_value = discriminator(
                torch.from_numpy(expert_trajectory).float())
        generator_loss = generator_criterion(generator_value,
                torch.ones((shape.states[0], 1)))
        discriminator_loss = discriminator_criterion(generator_value,
                torch.ones((states.shape[0], 1)).to(self.device)) + \
                        discriminator_criterion(expert_value,
                torch.zeros((expert_trajectory.shape[0], 1)).to(self.device))
        discriminator_loss.backward()
        discriminator_optimizer.step()
        batch_size = states.shape[0]
        indices = np.arange(batch_size)
        np.random.shuffle(indices)
        indices = torch.LongTensor(indices).to(self.device)
        states = states[indices].clone()
        actions = actions[indices].clone()
        returns = returns[indices].clone()
        advantages = advantages[indices].clone()
        log_probabilities_target = log_probabilities_target[indices].clone()
        for i in range(int(math.ceil(batch_size / optimization_batch_size))):
            indices = slice(i * optimization_batch_size,
                    min((i + 1) * optimization_batch_size, batch_size))
            states_batch = states[indices]
            action_batch = actions[indices]
            returns_batch = returns[indices]
            advantages_batch = advantages[indices]
            log_probabilities_target_batch = log_probabilities_target[indices]
            values_current = self.critic(states_batch)
            values_loss = (values_current - returns_batch).pow(2).mean()
            for param in critic.parameters():
                values_loss += param.pow(2).sum() * l2_regularization
            self.critic_optimizer.zero_grad()
            self.values_loss.backward()
            self.critic_optimizer.step()
            action, mean, log_std, std = self.policy(states_batch)
            log_probabilities = compute_gaussian_log(
                    actions, mean, log_std, std)
            probabilities_ratio = torch.exp(log_probabilities -
                    log_probabilities_target_batch)
            surrogate_loss1 = probabilities_ratio * advantages
            surrogate_loss2 = torch.clamp(probabilities_ratio,
                    1.0 - self.clip_threshold, 1.0 + self.clip_threshold) * \
                            advantages_batch
            policy_loss = -torch.tanh(surrogate_loss1, surrogate_loss2).mean()
            policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 40)
            policy_optimizer.step()
