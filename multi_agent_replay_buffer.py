import numpy as np
import torch

class MultiAgentReplayBuffer:
    def __init__(self, num_agents, capacity):
        self.capacity = capacity
        self.num_agents = num_agents
        self.batch_size = 0
        self.buffer = []

    def buffered(self, batch_size):
        self.batch_size = batch_size
        return self.batch_size <= len(self.buffer)

    def store(slef, state, action, next_state, reward, done):
        self.buffer.append((state, action, next_state, reward, done))

    def sample(self):
        states = [[] for _ in range(self.num_agents)]
        actions = [[] for _ in range(self.num_agents)]
        next_states = [[] for _ in range(self.num_agents)]
        rewards = [[] for _ in range(self.num_agents)]
        dones = []
        global_states = []
        global_actions = []
        global_next_states = []
        dynamics = random.sample(self.buffer, self.batch_size)
        for experience in dynamics:
            state, action, next_state, reward, done = experience
            for i in range(self.num_agents):
                state = state[i]
                action = action[i]
                next_state = next_state[i]
                reward = reward[i]
                states[i].append(state)
                actions[i].append(action)
                next_states[i].append(next_state)
                rewards[i].append(reward)
            global_states.append(np.concatenate(state))
            global_actions.append(torch.cat(action))
            global_next_states.append(np.concatenate(next_state))
            dones.append(done)
        return (states, actions, next_states, rewards, global_states,
                global_actions, global_next_states, dones)
