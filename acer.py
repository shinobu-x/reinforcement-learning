import numpy as np
import torch
import torch.nn as nn

# Sample Efficient Actor-Critic with Experience Replay
# https://arxiv.org/abs/1611.01224
class Backbone(nn.Module):
    def __init__(self, input_size, hidden_size, enable_recurrent):
        super(Backbone, self).__init__()
        self.hidden_size = hidden_size
        self.recurrent = enable_recurrent
        if recurrent:
            self.gru = nn.GRU(input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'base' in name:
                    nn.init.constant_(param, 0)
                else:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self.recurrent

    @property
    def hidden_size(self):
        if self.recurrent:
            return self.hidden_size
        else:
            return 1

    @property
    def output_size(self):
        return self.hidden_size

    def forward_gru(self, x, h_x, mask):
        if x.size(0) == h_x.size(0):
            x, h_x = self.gru(x.unsqueeze(0), (h_x * mask).unsqueeze(0))
            x = x.squeeze(0)
            h_x = h_x.squeeze(0)
        else:
            n = h_x.size(0)
            t = int(x.size(0) / n)
            mask = mask.view(t, n)
            zeros = (mask[1: ] == 0.0).any(dim = -1).nonzero().squeeze().cpu()
            if zeros.dim() == 0:
                zeros = [zeros.item() + 1]
            else:
                zeros = (zeros + 1).numpy().tolist()
            zeros = [0] + zeros + [t]
            h_x = h_x.unsqueeze(0)
            outputs = []
            for i in range(len(zeros) - 1):
                start = zeros[i]
                end = zeros[i + 1]
                scores, h_x = self.gru(x[start: end],
                        h_x * mask[start].view(1, -1, 1))
                outputs.append(scores)
            x = torch.cat(outputs, dim = 0)
            x = x.view(t * n, -1)
            h_x = h_x.squeeze(0)

class Actor(Backbone):
    def __init__(self, state_space, action_space, hidden_size,
            enable_recurrent = False):
        super(Actor, self).__init__(state_space, hidden_size,
                enable_recurrent)
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.enable_recurrent = enable_recurrent
        if self.enable_recurrent:
            state_space = hidden_size
        self.l1 = nn.Linear(state_space, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, state, h_x, mask):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        if self.enable_recurrent:
            x, h_x, = self.forward_gru(state, h_x, mask)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return x, h_x

class Critic(Backbone):
    def __init__(self, state_space, action_space, hidden_size,
            enable_recurrent = False):
        super(Critic, self).__init__(state_space, hidden_size, enable_recurrent)
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.enable_recurrent = enable_recurrent
        self.l1 = nn.Linear(state_space, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, action_space)

    def forward(self, x, h_x, mask):
        if self.enable_recurrent:
            x, h_x = self.forward_gru(x, h_x, mask)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class Agent(nn.Module):
    def __init__(self, state_space, action_space, hidden_size,
            enable_recurrent = False):
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.enable_recurrent = enable_recurrent

class ACER(object):
    def __init__(self, actor, critic, on_rollouts, off_rollouts, replay_buffer,
            episode_rewards, env):
        self.actor = actor
        self.critic = acritic
        self.on_rollouts = on_rollouts
        self.off_rollouts = off_rollouts
        self.replay_buffer = replay_buffer
        self.env = env
        self.num_timesteps = on_rollouts.num_timesteps
        self.device = self.on_rollouts.obs.device

        def train(self, enable_on_policy):
            rollouts = self.on_rollouts if enable_on_policy else off_policy
            if enable_on_policy:
                for t in range(self.num_timesteps):
                    with torch.no_grad():
                        probabilities, _, q_value, action, _, h_x
