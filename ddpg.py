import torch
import torch.nn.functional as F
from torch import nn

# Continuous control with deep reinforcement learning
# https://arxiv.org/abs/1509.02971
class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.l1 = nn.Linear(self.state_space, 128)
        self.n1 = nn.LayerNorm(128)
        self.l2 = nn.Linear(128, 256)
        self.n2 = nn.LayerNorm(256)
        self.l3 = nn.Linear(256, self.action_space)

    def forward(self, states):
        x = F.relu(self.n1(self.l1(state)))
        x = F.relu(self.n2(self.l2(state)))
        return torch.tanh(self.l3(x))

class Critic(nn.Module):
    def __inti__(self, state_space, action_space):
        super(Critic, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.l1 = nn.Linear(self.state_space, 128)
        self.n1 = nn.LayerNorm(128)
        self.l2 = nn.Linear(128 + self.action_space, 256)
        self.n2 = nn.LayerNorm(256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, states, actions):
        x = F.relu(self.n1(self.l1(state_space)))
        x = torch.cat((x, actions), 1)
        x = F.relu(self.n2(self.l1(x)))
        return self.l3(x)

class DDPG(object):
    def __init__(self, gamma, tau, state_space, action_space):
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.gamma = gamma
        self.tau = tau
        self.policy = Actor(state_space, action_space).to(self.device)
        self.policy_target = Actor(state_space, action_space).to(self.device)
        self.critic = Critic(state_space, action_space).to(self.device)
        self.critic_target = Critic(state_space, action_space).to(self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr = 1e-4)
        self.critic_optimizer = Adam(self.critic.parameters(), lr = 1e-4)
        self.update_params(self.policy, self.policy_target)
        self.update_params(self.critic, self.critic_target)

    def update_parameters(self, source, target, is_soft = False):
        if is_soft:
            for source_param, target_param in zip(source.parameters(),
                    target.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) +
                        source_param.data * self.tau)
        else:
            for source_param, target_param in zip(source.parameters(),
                    target.parameters()):
                target_param.data.copy_(source_param.data)

    def select_action(self, state, action_noise = None):
        mu = self.policy(state.to(self.device))
        mu = mu.data
        if action_noise is not None:
            noise = torch.Tensor(action_noise.generate()).to(device)
            mu += noise
        return mu.clamp(self.action_space.low[0], self.action_space.high[0])

    def train(self, replay_buffer):
        states, actions, next_states, rewards, not_done = replay_buffer.sample()
        next_actions = self.policy_target(next_states)
        values_target = self.critic_target(next_states, next_actions)
        rewards = rewards.unsqueeze(1)
        not_done = not_done.unsqueeze(1)
        expected_values = reward + self.gamma * next_values * not_done
        self.critic_optimizer.zero_grad()
        values_current = self.ciritc(states, actions)
        critic_loss = F.mse_loss(values_current, values_target)
        ciritc_loss.backward()
        self.critic_optimizer.step()
        self.policy_optimizer.zero_grad()
        policy_loss = -self.critic(states, self.actor(states))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.policy_optimizer.step()
        self.update_parameters(self.policy, self.policy_target, is_soft = True)
        self.update_parameters(self.critic, self.ciritc_target, is_soft = True)
        return ciritc_loss.item(), policy_loss.item()
