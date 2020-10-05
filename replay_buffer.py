import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, state_space, action_space, capacity):
        self.capacity = capacity
        self.buffer = 0
        self.position = 0
        self.batch_size = 0
        self.state = np.zeros((capacity, state_space))
        self.action = np.zeros((capacity, action_space))
        self.reward = np.zeros((capacity, 1))
        self.next_state = np.zeros((capacity, state_space))
        self.not_done = np.zeros((capacity, 1))
        self.device = torch.device('cuda'
                if torch.cuda.is_available() else 'cpu')

    def buffered(self, batch_size):
        self.batch_size = batch_size
        return batch_size <= self.position

    def store(self, state, action, next_state, reward, not_done):
        self.state[self.position] = state
        self.action[self.position] = action
        self.next_state[self.position] = next_state
        self.reward[self.position] = reward
        self.not_done[self.position] = 1. - not_done
        self.position = (self.position + 1) % self.capacity
        self.buffer = min(self.buffer + 1, self.capacity)

    def sample(self):
        indices = np.random.randint(0, self.buffer, size = self.batch_size)
        return (torch.FloatTensor(self.state[indices]).to(self.device).
                unsqueeze(1),
                torch.LongTensor(self.action[indices]).to(self.device).
                unsqueeze(1),
                torch.FloatTensor(self.next_state[indices]).to(self.device).
                unsqueeze(1),
                torch.FloatTensor(self.reward[indices]).to(self.device),
                torch.FloatTensor(self.not_done[indices]).to(self.device))
