import numpy as np
from mdp import MDP

class SinglePathMDP(MDP):
    def __init__(self, action_space, horizon, enable_stochastic = True):
        super(SinglePathMDP, self).__init__(horizon, action_space)
        self.rewards = np.ones((horizon, action_space)) * np.array([1, 0])
        self.state = 0.0
        self.timestep = 0
        self.initial_state = 0.0
        self.dynamics = np.zeros((horizon, action_space, horizon + 1))
        probability = np.zeros(action_space)
        if enable_stochastic:
            probability = np.arange(action_space) / float(action_space)
        for time in range(horizon):
            for action in range(action_space):
                self.dynamics[time, action, time + 1] = \
                        1.0 - probability[action]
                self.dynamics[time, action, time] = probability[action]

    def step(self, action):
        state, reward, _ = super(SinglePathMDP, self).step(action)
        self.timestep += 1
        done = self.timestep == self.horizon
        return state, reward, done

    def reset(self):
        self.timestep = 0
        self.state = self.initiail_state
        return self.state

    def get_policy(self, tau):
        probability = 0.5
        if tau == 0: probability = 0.4
        if tau == 1: probability = 0.6
        q = (1 - probability) / (self.action_space - 1)
        multiplier = np.array([probability] +
                [q for _ in range(self.action_space - 1)])
        policy = np.ones((self.horizon, self.action_space)) * multiplier
        return policy
