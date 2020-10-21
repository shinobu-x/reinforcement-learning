import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy

# Asynchronous Methods for Deep Reinforcement Learning
# https://arxiv.org/abs/1602.01783
class Agent(nn.Module):
    def __init__(self, state_space, action_space):
        super(Agent, self).__init__()
        self.conv1 = nn.Conv2d(state_space, 32, 5, stride = 1, padding = 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride = 1, padding = 1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.lstm = nn.LSTMCell(1024, 512)
        self.actor = nn.Linear(512, action_space)
        self.critic = nn.Linear(512, 1)

    def forward(self, state, hx, cx):
        x = F.relu(self.max_pool(self.conv1(state)))
        x = F.relu(self.max_pool(self.conv2(x)))
        x = F.relu(self.max_pool(self.conv3(x)))
        x = F.relu(self.max_pool(self.conv4(x)))
        x = x.view(x.size(0), -1)
        hx, cx = self.lstm(x, (hx, cx))
        logit = self.actor(hx)
        value = self.critic(hx)
        return value, logit, (hx, cx)

class Learner(object):
    def __init__(self, model, env, state):
        self.model = model
        self.env = env
        self.state = state
        self.hx = None
        self.cx = None
        self.epsilon = 0.0
        self.values = []
        self.log_probabilities = []
        self.rewards = []
        self.entropies = []
        self.done = True
        self.info = None
        self.reward = 0.0
        self.id = -1

    def act(self, state, hx, cx):
        self.state = state.unsqueeze(0)
        value, logit, (self.hx, self.cx) = self.model(self.state, self.hx,
                self.cx)
        probability = F.softmax(logit, dim = 1)
        log_probability = F.log_softmax(logit, dim = 1)
        entropy = -(probability * log_probability).sum(1)
        self.entropies.append(entropy)
        action = probability.multinomial(1).data
        log_probability = log_probability.gather(1,
                torch.autograd.Variable(action))
        action = action.cpu().numpy()
        state, self.reward, self.done, _ = self.env.step(action)
        self.state = torch.from_numpy(state).float()
        self.reward = max(min(self.reward, 1), -1)
        self.values.append(value)
        self.log_probabilities.append(log_probability)
        self.rewards.append(self.reward)

    def clear(self):
        self.values = []
        self.log_probabilities = []
        self.rewards = []
        self.entropies = []
        return self

class A3C(object):
    def __init__(self, env):
        self.device = torch.device('cuda' if torch.cuda.is_available() \
                else 'cpu')
        self.env = env
        self.steps = 1000
        self.gamma = 1e-2
        self.tau = 1e-2
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.shared_model = Agent(self.state_space, self.action_space).to(
                self.device)
        self.shared_model_optimizer = optim.Adam(self.shared_model.parameters(),
                lr = 3e-4, weight_decay = 1e-1)

    def train(self, rank):
        learner = Learner(None, self.env, None)
        learner.model = Agent(self.state_space, self.action_space).to(
                self.device)
        learner.state = learner.env.reset()
        learner.state = torch.from_numpy(learner.state).float().to(self.device)
        learner.model = deepcopy(self.shared_model)
        while True:
            if learner.done:
                learner.hx = torch.autograd.Variable(torch.zeros(1, 512)).to(
                        self.device)
                learner.cx = torch.autograd.Variable(torch.zeros(1, 512)).to(
                        self.device)
            else:
                learner.hx = torch.autograd.Variable(learner.hx.data).to(
                        self.device)
                learner.cx = torch.autograd.Variable(learner.cx.data).to(
                        self.device)
            for step in range(self.steps):
                learner.act(learner.state, learner.hx, learner.cx)
                if learner.done:
                    break
            if learner.done:
                state = learner.env.reset()
                learner.state = torch.from_numpy(state).float().to(self.device)
            R = torch.zeros(1, 1)
            if not learner.done:
                state = torch.autograd.Variable(learner.state.unsqueeze(0)).to(
                        self.device)
                value, _, _ = learner.model(state, (learner.hx, learner.cx))
                R = torch.autograd.Variable(value.data).to(self.device)
            learner.values.append(R)
            policy_loss = 0.0
            value_loss = 0.0
            GAE = torch.zeros(1, 1).to(self.device)
            for i in reversed(range(len(learner.rewards))):
                R = self.gamma * R + learner.rewards[i]
                A = R - learner.values[i]
                value_loss = value_loss + 0.5 * A.pow(2)
                delta = learner.rewards[i] + self.gamma * \
                        learner.values[i + 1].data - learner.values[i].data
                GAE = GAE * self.gamma * self.tau + delta
                policy_loss = policy_loss - learner.log_probabilities[i] * \
                        torch.autograd.Variable(GAE) - 1e-2 * \
                        learner.entropies[i]
            loss = policy_loss + 0.5 * value_loss
            learner.model.zero_grad()
            loss.backward()
            for param, shared_param in zip(learner.model.parameters(),
                    self.shared_model.parameters()):
                if shared_param.grad is None:
                    shared_param._grad = param.grad.cpu()
            self.shared_model_optimizer.step()
            learner.clear()
