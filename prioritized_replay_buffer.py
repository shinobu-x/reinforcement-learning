import gym
import numpy as np
import random
import torch
from collections import namedtuple
from itertools import count

# Prioritized Experience Replay
# https://arxiv.org/abs/1511.05952
class SumTree:
  '''
  Every node is the sum of its children, with the priorities as the leaf nodes
  '''
  write = 0
  def __init__(self, capacity):
    self.capacity = capacity
    self.tree = np.zeros(2 * capacity - 1)
    self.data = np.zeros(capacity, dtype = object)
    self.index_leaf_start = capacity - 1

  def _propagate(self, index, change):
    parent = (index - 1) // 2
    self.tree[parent] += change
    if parent != 0:
      self._propagate(parent, change)

  def _retrieve(self, index, sum):
    left = 2 * index + 1
    right = left + 1
    if left >= len(self.tree):
      return index
    if sum <= self.tree[left]:
      return self._retrieve(left, sum)
    else:
      return self._retrieve(right, sum - self.tree[left])

  def total(self):
    return self.tree[0]

  def max(self):
    return self.tree[self.index_leaf_start:].max()

  def add(self, parent, dynamics):
    index = self.write + self.index_leaf_start
    self.data[self.write] = dynamics
    self.update(index, parent)
    self.write += 1
    if self.write >= self.capacity:
      self.write = 0

  def update(self, index, parent):
    change = parent - self.tree[index]
    self.tree[index] = parent
    self._propagate(index, change)

  def get(self, sum):
    index = self._retrieve(0, sum)
    data_index = index - self.capacity + 1
    return (index, self.tree[index], self.data[data_index])

class PrioritizedReplayBuffer:
  def __init__(self, num_episodes, capacity, epsilon = 0.0001, alpha = 0.5,
      beta = 0.4, index = 0):
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.dynamics = namedtuple('Dynamics', ('state', 'action', 'next_state',
        'reward', 'not_done'))
    self.num_episodes = num_episodes
    self.episode = 0
    self.batch_size = 0
    self.capacity = capacity
    self.epsilon = epsilon
    self.alpha = alpha
    self.beta = beta
    self.tree = SumTree(capacity)
    self.last_index = 0

  def _get_priority(self, delta):
    # delta: TD-error
    return (torch.abs(td_error) * self.epsilon) ** self.alpha

  def buffered(self, batch_size):
      self.batch_size = batch_size
      return batch_size <= self.last_index

  def store(self, *args):
    self.last_index += 1
    priority = self.tree.max()
    if priority <= 0:
      priority = 1
    self.tree.add(priority, *args)

  def sample(self):
    self.episode += 1
    trajectories = []
    weights = np.empty(self.batch_size, dtype = 'float32')
    total_probabilities = self.tree.total()
    # beta annealing
    beta = self.beta + (1 - self.beta) * self.episode / self.num_episodes
    beta = min(1.0, beta)
    for i, sum in enumerate(np.random.uniform(0, total_probabilities,
      self.batch_size)):
      index, priority, dynamics = self.tree.get(sum)
      trajectories.append(dynamics)
      # Compute importance-sampling weight
      weights[i] = \
          (self.capacity * priority / total_probabilities) ** (-beta)
      weight = weights / weights.max()
    dynamics = [trajectories[i] for i in range(self.batch_size)]
    dynamics = self.dynamics(*zip(*trajectories))
    return (torch.FloatTensor(dynamics.state),
            torch.LongTensor(dynamics.action).unsqueeze(1),
            torch.FloatTensor(dynamics.next_state),
            torch.FloatTensor(dynamics.reward),
            torch.FloatTensor(dynamics.not_done),
            weight)

  def update(self, index, delta):
    # delta: TD-error
    priority = self._getPriority(delta)
    self.tree.update(index, priority)

  def __len__(self):
    return self.last_index
