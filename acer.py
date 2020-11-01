import torch
from torch import nn

class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_space):
        super(Actor).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_space = hidden_space
        self.l1 = nn.Linear(self.state_space, 128)
        self.l2 = nn.Linear(128, self.hidden_space)
        self.lstm = nn.LSTMCell(self.hidden_space, self.hidden_space)
        self.l3 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, state, hidden_state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        hidden_state = self.lstm(x, hidden_state)
        x = hidden_state[0]
        distribution = F.softmax(self.l3(x), dim = 1).clamp(max = 1 - 1e-20)
        return distribution, hidden_state

class Critic(nn.Module):
    def __init__(self, state_space, action_space, hidden_space):
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_space = hidden_space
        self.l1 = nn.Linear(self.state_space, 128)
        self.l2 = nn.Linear(128, self.hidden_space)
        self.lstm = nn.LSTMCell(self.hidden_space, self.hidden_space)
        self.l3 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, state, hidden_state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        hidden_state = self.lstm(x, hidden_state)
        x = hidden_state[0]
        q = self.l3(x)
        return q # compute v later

class Agent:
    def __init__(self, state_space, action_space, hidden_space):
        self.actor = Actor(state_space, action_space, hidden_space)
        self.critic = Critic(state_space, action_space, hidden_space)

    def forward(self, state, hidden_state):
        distribution, hidden_state = self.actor(state, hidden_state)
        Q = self.critic(state, hidden_state)
        V = (Q * distribution).sum(1, keepdim = True)
        return distribution, Q, V, hidden_state

class ACER(object):
    def __init__(self, shared_model, shared_averaged_model, shared_optimizer,
            state_space, action_space, hidden_space, timesteps = 100000,
            done = True, forward_step = 100):
        self.device = torch.device('gpu' if torch.cuda.is_available()
                else 'cpu')
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_space = hidden_space
        self.agent = Agent(self.state_space, self.action_space,
                self.hidden_space).to(self.device)
        self.timesteps = timesteps
        self.done = done
        self.device = torch.device('gpu' if torch.cuda.is_available()
                else 'gpu')
        self.forward_steps = forward_steps

    def train(self):
        current_timestep = 1
        while current_timestep <= self.timesteps:
            agent.load_state_dict(shared_model.state_dict())
            if self.done:
                hidden_state = torch.zeros(1, hidden_space)
                average_hidden_state = torch.zeros(1, hidden_space)
                current_state = torch.zeros(1, hidden_space)
                average_current_state = torch.zeros(1, hidden_space)
                state = torch.from_numpy(state).float().unsqueeze(0).to(
                        self.device)
            else:
                hidden_state = hidden_state.detach()
                current_state = current_state.detach()
            distributions = []
            Qs = []
            Vs = []
            actions = []
            rewards = []
            average_distributions = []
            while not done and current_timestep <= self.forward_steps:
                distribution, Q, V, (hidden_state, current_state) = \
                        agent(state, (hidden_state, current_state))
