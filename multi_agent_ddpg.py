import random
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

# Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
# https://arxiv.org/abs/1706.02275
class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.l1 = nn.Linear(self.state_space, 256)
        self.l2 = nn.Linear(256, 512)
        self.l3 = nn.Linear(512, self.action_space)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x

class Critic:
    def __init__(self, state_space, action_space):
        super(Critic, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.l1 = nn.Linear(self.state_space, 256)
        self.l2 = nn.Linear(256 + self.action_space, 512)
        self.l3 = nn.Linear(512, 1)

    def forward(self, state, action):
        x = F.relu(self.l1(state))
        x = torch.cat([x, a], 1)
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class Agent:
    def __init__(self, env, agent_id, gamma = 0.99, tau = 1e-2, epsilon = 1e-3):
        self.env = env
        self.agent_id = agent_id
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.device = torch.device('gpu' if torch.cuda.is_available()
                else 'cpu')
        self.state_space = self.env.observation_space[self.agent_id].shape[0]
        self.action_space = self.env.action_space[self.agent_id].n
        self.num_agents = self.env.n
        self.state_spaces = int(np.sum([self.env.state_space[i].shape[0]
            for i in range(self.env.n)]))
        self.policy = Actor(self.state_space, self.action_space).to(self.device)
        self.policy_target = Actor(self.state_space, self.action_space).to(
                self.device)
        self.policy_optimizer = optim.Adam(self.actor.parameters(), lr = 1e-2)
        self.critic = Critic(self.state_spaces,
                self.action_space * self.num_agents).to(self.device)
        self.critic_target = Critic(self.state_spaces, self.action_space *
                self.num_agents).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = 1e-2)
        self.update_target()

    def select_action(self, state):
        x = torch.autograd.Variable(
                torch.from_numpy(state).float().squeeze(0)).to(self.device)
        x = self.policy(x)
        x = self.compute_onehot(x)
        return x

    def compute_onehot(self, logits, epsilon = 0.0):
        argmax_a = (logits == logits.max(0, keepdim = True)[0]).float()
        if epsilon == 0.0:
            return argmax_a
        x = torch.autograd.Variable(torch.eye(
            logits.shape[1])[[np.random.choice(range(logits.shape[1]),
                size = logits.shape[0])]], requires_grads = False)
        return torch.stack([argmax_a[i] if r > epsilon else x[i] for i, r in
            enumerate(torch.rand(logits.shape[0]))])

    def update(self, states, rewards, global_states, global_actions,
            global_next_states, global_next_actions):
        self.critic_optimizer.zero_grad()
        Q_current = self.critic(global_states, global_actions)
        Q_target = self.critic_target(global_next_states, global_next_actions)
        Q_target = (rewards + self.gamma * Q_target).detach()
        critic_loss = F.mse_loss(Q_current, Q_target)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        self.policy_optimizer.zero_grad()
        policy_loss = -self.critic(global_states, global_actions).mean()
        policy_current = self.policy(states)
        policy_loss += -(policy_current ** 2).mean() * 1e-3
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.policy_optimizer.step()

    def update_target(self):
        for param, param_target in zip(self.actor,parameters(),
                self.actor.parameters()):
            param_target.data.copy_(param.data)
        for param, param_target in zip(self.critic.parameters(),
                self.critic_target.parameters()):
            param_target.data.copy_(param.data)

class MultiAgentDDPG:
    def __init__(self):
        self.num_agents = None
        self.agents = None
        self.batch_size = None
        self.device = torch.device('gpu' if torch.cuda.is_available()
                else 'cpu')

    def select_actions(self, states):
        actions = []
        for i in range(self.num_agents):
            action = self.agents[i].select_action(states[i])
            actions.append(action)
        return actions

    def update(self, replay_buffer):
        states, actions, next_states, rewards, global_states, lobal_actions, \
                global_next_states, dones = replay_buffer.sample(
                        self.batch_size)
        for i in range(self.num_agents):
            state = states[i]
            action = actions[i]
            next_state = next_states[i]
            reward = rewards[i]
            global_next_actions = []
            for agent in self.agents:
                next_state = torch.FloatTensor(next_state).to(self.device)
                next_action = agent.policy(next_state)
                next_action = [agent.compute_onehot(next_action_i)
                        for next_action_i in next_action]
                next_action = torch.stack(next_action)
                global_next_actions.append(next_action)
            global_next_actions = torch.cat([next_action_i
                for next_action_i in global_next_action], 1)
            state = torch.FloatTensor(state).to(self.device)
            reward = torch.FloatTensor(reward).view(reward.size(0), 1).to(
                    self.device)
            global_states = torch.FloatTensor(global_states).to(self.device)
            global_actions = torch.stack(global_actions).to(self.device)
            global_next_states = torch.FloatTensor(global_next_states).to(
                    self.device)
            self.agents[i].update(state, reward, global_states, global_actions,
                global_next_states)
            self.agents[i].update_target()
