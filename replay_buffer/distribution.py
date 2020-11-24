class ReplayBuffer:
    def __init__(self, action_space, num_agents, capacity):
        self.action_space = action_space
        self.num_agents = num_agents
        self.actions = []
        self.states = []
        self.distribution = [[] for _ in range(num_agents)]
        self.rewards = []
        self.not_done = [[] for _ in range(num_agents)]

    def buffered(self, batch_size):
        return batch_size < len(self.actions)

    def sample(self):
        actions = torch.tensor(self.actions)
        states = self.states
        distribution = []
        for i in range(self.num_agents):
            distribution.append(torch.cat(self.distribution[i]).view(
                len(self.distribution[i]), self.action_space))
        reward = torch.tensor(self.reward)
        not_done = self.not_done
        return actions, states, distribution, reward, not_done
