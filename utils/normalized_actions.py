import gym

class NormalizedAction(gym.ActionWrapper):
    def compute_normalization(self, action):
        action = (action + 1) / 2
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def compute_reverse_normalization(self, action):
        action -= self.action_space.low
        action /= (self.action_space.hight - self.action_space.low)
        action = action * 2 - 1
        return action
