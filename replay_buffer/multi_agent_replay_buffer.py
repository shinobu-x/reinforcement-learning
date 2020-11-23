import numpy as np
import random
import torch

class MultiAgentReplayBuffer(object):
    def __init__(self, state_space, action_space, num_agents, capacity):
        self.capacity = capacity
        self.batch_size = 0
        self.num_agents = num_agents
        self.buffer = []

    def buffered(self, batch_size):
        self.batch_size = batch_size
        return batch_size <= len(self.buffer)

    def store(self, state, action, next_state, reward, not_done):
        self.buffer.append((state, action, np.array(reward), next_state,
            not_done))

    def sample(self):
        states = [[] for _ in range(self.num_agents)]
        actions = [[] for _ in range(self.num_agents)]
        next_states = [[] for _ in range(self.num_agents)]
        rewards = [[] for _ in range(self.num_agents)]
        global_states = []
        global_actions = []
        global_next_states = []
        dones = []
        batch = random.sample(self.buffer, self.batch_size)
        for experience in batch:
            state, action, next_state, reward, not_done = experience
            for i in range(self.num_agents):
                state_i = state[i]
                action_i = action[i]
                next_state_i = next_state[i]
                reward_i = reward[i]
                states[i].append(state_i)
                actions[i].append(action_i)
                next_states[i].append(next_state_i)
                rewards[i].append(reward_i)
            global_states.append(np.concatenate(state))
            global_actions.append(np.concatenate(action))
            global_next_states.append(np.concatenate(next_state))
        return states, actions, next_states, rewards, global_states, \
                global_actions, global_next_states
