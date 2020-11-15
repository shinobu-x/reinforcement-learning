import torch
import torch.nn.functional as F
from torch import nn
from distributions.categorical import Categorical
from modules.gru import GRU

# DiCE: The Infinitely Differentiable Monte-Carlo Estimator
# https://arxiv.org/abs/1802.05098
class Actor(nn.Module):
    def __init__(self, state_space, hidden_size):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_space, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, states):
        return  F.relu(self.l2(F.relu(self.l1(states))))

class Critic(nn.Module):
    def __init__(self, state_space):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_space, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

    def forward(self, stetes):
        x = F.relu(self.l2(F.relu(self.l1(states))))
        return self.l3(x), x

class Discriminator(nn.Module):
    def __init__(self, state_space, action_space):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(state_space + action_space, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim = -1)
        return self.l3(F.relu(self.l2(F.relu(self.l1(x)))))

class Base(nn.Module):
    def __init__(self, state_space, enable_recurrent = False, hidden_size = 64):
        super(Base, self).__init__(state_space, enable_recurrent, hidden_size)
        self.enable_recurrent = enable_recurrent
        self.actor = Actor(state_space, hidden_size)
        self.critic = Critic(state_space, hidden_size)
        self.gru = GRU(state_space, hidden_size)

    def forward(self, states, hxs, masks):
        if self.enable_recurrent:
            states, hxs = self.gru(states, hxs, masks)
        actor_hidden_state = self.actor(states)
        value, critic_hidden_state = self.critic(states)
