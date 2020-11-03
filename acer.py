import torch
from torch import nn

class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_space):
        super(Actor, self).__init__()
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
        super(Critic, self).__init__()
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

class Agent(nn.Module):
    def __init__(self, state_space, action_space, hidden_space):
        super(Agent, self).__init__()
        self.actor = Actor(state_space, action_space, hidden_space)
        self.critic = Critic(state_space, action_space, hidden_space)

    def forward(self, state, hidden_state):
        distribution, hidden_state = self.actor(state, hidden_state)
        Q = self.critic(state, hidden_state)
        V = (Q * distribution).sum(1, keepdim = True)
        return distribution, Q, V, hidden_state

class ACER(object):
    def __init__(self, env, shared_model, shared_average_model,
            shared_optimizer, state_space, action_space, hidden_space,
            replay_buffer, num_timesteps = 100000, done = True,
            num_forward_steps = 100, num_episodes = 100):
        self.env = env
        self.device = torch.device('gpu' if torch.cuda.is_available()
                else 'cpu')
        self.shared_model = shared_model
        self.shared_average_model = shared_average_model
        self.shared_optimizer = shared_optimizer
        self.state_space = state_space
        self.action_space = action_space
        self.hidden_space = hidden_space
        self.replay_buffer = replay_buffer
        self.agent = Agent(self.state_space, self.action_space,
                self.hidden_space).to(self.device)
        self.num_timesteps = num_timesteps
        self.done = done
        self.num_forward_steps = num_forward_steps
        self.num_episodes = num_episodes

    def train(self):
        current_timestep = 1
        while current_timestep <= self.timesteps:
            agent.load_state_dict(shared_model.state_dict())
            if self.done:
                hidden_state = torch.zeros(1, hidden_space)
                average_hidden_state = torch.zeros(1, hidden_space)
                current_state = torch.zeros(1, hidden_space)
                average_current_state = torch.zeros(1, hidden_space)
                state = torch.from_numpy(self.env.reset())
                self.done = False
                current_episode = 0
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
                average_distribution, _, _, (average_hidden_state,
                        average_current_state) = shared_average_model(state,
                                (average_hidden_state, average_current_state))
                action = torch.multinomial(distribution, 1)[0, 0]
                next_state, reward, done, _ = env.step(action.item())
                next_state = torch.from_numpy(next_state).float().to(
                        self.device)
                reward = min(max(reward, -1), 1)
                self.done = self.done or current_episode >= self.num_episodes
                current_episode += 1
                if not self.enable_on_policy:
                    self.replay_buffer.store(state, action, reward,
                            distribution.detach())
                distributions.append(distribution)
                Qs.append(Q)
                Vs.append(V)
                actions.append(torch.LongTensor(action))
                rewards.append(torch.Tensor(reward))
                average_distributions.append(average_distribution)
                current_timestep += 1
                if done:
                    Q = torch.zeros(1, 1)
                    if not self.enable_on_policy:
                        self.replay_buffer.store(state, None, None, None)
                else:
                    _, _, Q, _ = agent(state, (hidden_state, current_state))
                    Q = Q.detach()

