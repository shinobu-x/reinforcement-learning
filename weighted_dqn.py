import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

# Human-level control through deep reinforcement learning
# https://www.nature.com/articles/nature14236
class Agent(nn.Module):

    def __init__(self, state_space, action_space):
        super(Agent, self).__init__()
        self.conv1 = nn.Conv2d(state_space[0], 16, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2)
        self.l1 = nn.Linear(2592, 256)
        self.l2 = nn.Linear(256, action_space)

    def forward(self, state):
        a = F.relu(self.conv1(state))
        a = F.relu(self.conv2(a))
        a = F.relu(self.l1(a.view(a.size(0), -1)))
        a = F.softmax(self.l2(a))
        return a

class WeightedDQN(object):
    def __init__(self, state_space, action_space, gamma = 0.99):
        self.action_space = action_space
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.policy = Agent(state_space, action_space).to(self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr = 3e-4,
                weight_decay = 1e-1)
        self.epsilon = 0.09
        self.gamma = gamma

    def select_action(self, state):
        #state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        state = torch.autograd.Variable(state, requires_grad = True)
        distribution = self.policy(state)
        # action = distribution.max(1)[1].view(1, 1).detach().item()
        action = distribution.multinomial(1).data
        log_probability = distribution[:, action[0, 0]].view(1, -1)
        return action, log_probability

    def train(self, replay_buffer):
        state, action, log_probability, next_state, reward, done = \
                replay_buffer.sample()
        state = torch.tensor(state)
        action = torch.tensor(action)
        next_state = torch.tensor(next_state)
        reward = torch.tensor(reward)
        done = torch.ByteTensor(done)
        log_p = None
        log_q = None
        Q_current = log_p = self.policy(state)
        Q_current = Q_current.gather(0, action.unsqueeze(-1)).squeeze(-1)
        Q_target = self.policy(next_state)
        Q_target = Q_target.max(1)[0].detach()
        Q_target = reward + self.gamma * Q_target
        log_q = torch.from_numpy(log_probability.astype(np.float32))
        Z = torch.exp(log_p) / torch.exp(log_q).reshape(-1, 1)
        self.policy_optimizer.zero_grad()
        loss = F.mse_loss(Q_current.float(), Q_target.float())
        loss /= Z.sum()
        loss.backward()
        clip_grad_norm_(self.policy.parameters(), 40)
        self.policy_optimizer.step()
