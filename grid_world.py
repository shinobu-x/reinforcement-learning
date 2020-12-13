import numpy as np
import random
from collections import namedtuple
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

dynamics = namedtuple('Dynamics', ('state', 'action', 'next_state', 'reward'))
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def is_buffered(self, batch_size):
        return batch_size < len(self.memory)

    def store(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = dynamics(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

class DQN(nn.Module):
    def __init__(self, state_space, action_space):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(state_space, 100)
        self.l2 = nn.Linear(100, 500)
        self.l3 = nn.Linear(500, action_space)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.l3(x)

class GridWorld(object):
    def __init__(self):
        self.state = np.zeros((4, 4, 4))
        self.player = np.array([0, 0, 0, 1])
        self.wall = np.array([0, 0, 1, 0])
        self.pit = np.array([0, 1, 0, 0])
        self.goal = np.array([1, 0, 0, 0])

    def find(self, state, obj):
        for i in range(0, 4):
            for j in range(0, 4):
                if (state[i, j] == obj).all(): return i, j

    def init_grid(self):
        state = self.state
        state[0, 1] = self.player
        state[2, 2] = self.wall
        state[1, 1] = self.pit
        state[3, 3] = self.goal
        return state

    def init_player(self, random = False):
        state = self.state
        state[(np.random.randint(0, 4), np.random.randint(0, 4))] = self.player
        if random:
            state[(np.random.randint(0, 4), np.random.randint(0, 4))] = \
                    self.wall
            state[(np.random.randint(0, 4), np.random.randint(0, 4))] = \
                    self.pit
            state[(np.random.randint(0, 4), np.random.randint(0, 4))] = \
                    self.goal
        else:
            state[2, 2] = self.wall
            state[1, 1] = self.pit
            state[1, 2] = self.goal
        player = self.find(state, self.player)
        wall = self.find(state, self.wall)
        pit = self.find(state, self.pit)
        goal = self.find(state, self.goal)
        return init_player() \
                if (not player or not wall or not pit or not goal) else state

    def move(self, state, action):
        player = self.find(state, self.player)
        wall = self.find(state, self.wall)
        pit = self.find(state, self.pit)
        goal = self.find(state, self.goal)
        state = self.state
        actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        pos = (player[0] + actions[action][0], player[1] + actions[action][1])
        if pos != wall:
            if (np.array(pos) <= (3, 3)).all() and \
                    (np.array(pos) >= (0, 0)).all():
                state[pos][3] = 1
        if not self.find(state, self.player):
            state[player] = self.player
        state[wall][2] = 1
        state[pit][1] = 1
        state[goal][0] = 1
        return state

    def get_pos(self, state, index):
        for i in range(0, 4):
            for j in range(0, 4):
                if state[i, j][index] == 1: return i, j

    def get_reward(self, state):
        player = get_pos(state, 3)
        pit = get_pos(state, 1)
        goal = get_pos(state, 0)
        if player == pit: return -10
        elif player == goal: return 10
        else: return -1

    def display_state(self, state):
        grid = np.zeros((4, 4), dtype = str)
        player = self.find(state, self.player)
        wall = self.find(state, self.wall)
        pit = self.find(state, self.pit)
        goal = self.find(state, self.goal)
        for i in range(0, 4):
            for j in range(0, 4):
                grid[i, j] = ''
        if player: grid[player] = 'P'
        if wall: grid[wall] = 'W'
        if pit: grid[pit] = '-'
        if goal: grid[goal] = '+'
        return grid

grid_world = GridWorld()
state = grid_world.init_grid()
#print(grid_world.display_state(state))
state = grid_world.init_player()
#print(grid_world.display_state(state))
state = grid_world.init_player(random = True)
#print(grid_world.display_state(state))
status = 1
dqn = DQN(64, 4)
state = Variable(torch.from_numpy(state).float())
state_value = dqn(state.view(-1, 64))
