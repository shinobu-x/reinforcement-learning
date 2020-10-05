import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from exploration_scheduler import ExplorationScheduler

# Human-level control through deep reinforcement learning
# https://www.nature.com/articles/nature14236
class Agent(nn.Module):
    def __init__(self, state_space, action_space):
        super(Agent, self).__init__()
        self.l1 = nn.Linear(state_space, 100)
        self.l2 = nn.Linear(100, 500)
        self.l3 = nn.Linear(500, action_space)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)
        return a

class DQN(object):
    def __init__(self, state_space, action_space, gamma = 0.99):
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.policy = Agent(state_space, action_space).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                weight_decay = 1e-1)
        self.exploration_scheduler = ExplorationScheduler(10000, 0.1)
        self.gamma = gamma

    def select_action(self, state, t):
        sample = random.random()
        threshold = self.exploration_scheduler.value(t)
        if 1000 < t:
            if sample > threshold:
                state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
                return self.policy(state).max(1)[1].view(1, 1).detach().item()
        return torch.tensor([[random.randrange(action_space)]],
                dtype = torch.long).item()

    def train(self, replay_buffer):
        state, action, next_state, reward, not_done = replay_buffer.sample()
        Q_current = self.policy(state)
        Q_current = Q_current.gather(0, action).squeeze(1)
        Q_target = self.policy(next_state)
        Q_target = Q_target.max(1)[0].detach()
        Q_target = reward + self.gamma * Q_target
        self.policy_optimizer.zero_grad()
        loss = F.mse_loss(Q_current.float(), Q_target.float())
        loss.backward()
        self.policy_optimizer.step()

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n # num_action env.action_space.shape[0]
    max_episode_steps = env._max_episode_steps
    policy = DQN(state_space, action_space)
    episode_reward = 0
    episode_timesteps = 0
    capacity = 100000
    use_replay_buffer = 1000
    batch_size = 32
    replay_buffer = ReplayBuffer(state_space, action_space, capacity)
    state, done = env.reset(), False
    for t in range(int(10000)):
        episode_timesteps += 1
        action = policy.select_action(state, t)
        #action = select_action(state)
        next_state, reward, done, _ = env.step(action)
        not_done = float(done) if episode_timesteps < max_episode_steps else 0
        episode_reward += reward
        replay_buffer.store(state, action, next_state, reward, not_done)
        state = next_state
        if replay_buffer.buffered(64):
            policy.train(replay_buffer)
