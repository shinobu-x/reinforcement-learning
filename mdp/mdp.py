import numpy as np

class MDP(object):
    def __init__(self, state_space, action_space, horizon):
        self.state_space = state_space
        self.action_space = action_space
        self.horiozn = horizon

    def reset(self):
        pass

    def step(self, action):
        states = np.arange(self.state_space + 1)
        num_states = np.random.choice(states,
                p = self.dynamics[self.state, action])
        reward = self.rewards[self.state, action]
        self.state = num_states
        return num_state, reward, False

    def get_policy(self, num):
        pass
