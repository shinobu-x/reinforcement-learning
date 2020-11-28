import torch
import torch.nn.functional as F
from copy import deepcopy
from torch import nn
from torch.optim import Adam

class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_space, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_space)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_space, action_space, num_agents):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_apce, 128)
        self.l2 = nn.Linear(128 + action_space, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = F.relu(self.l1(state))
        x = torch.cat([x, action], 1)
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class MADDPG(object):
    def __init__(self, state_space, action_space, num_agents, batch_size):
        self.state_space = state_space
        self.action_space = action_space
        self.num_agents = num_agents
        self.batch_size = batch_size
        self.policies = [Actor(state_space, action_space)
                for i in range(num_agents)]
        self.critics = [Critic(state_space, action_space, num_agents)
                for i in range(num_agents)]
        self.policies_target = deepcopy(self.policies)
        self.critics_target = deepcopy(self.critics)
        self.gamma = 0.95
        self.tau = 1e-2
        self.epsilon = 1e-2
        self.var = [1.0 for _ in range(num_agents)]
        self.policy_optimizers = [Adam(policy.parameters(), lr = 1e-4)
                for policy in self.policies]
        self.critic_optimizers = [Adam(critic.parameters(), lr = 1e-4)
                for critic in self.critics]
        self.step_counts = 0
        self.episode_counts = 0

    def update_parameters(self, source, target, is_soft = False):
        if is_soft:
            for source_param, target_param in \
                    zip(source.parameters(), target.parameters()):
                target.data.copy_(target_param.data * (1.0 - self.tau)  +
                        source_param.data * self.tau)
        else:
            for source_param, target_param in \
                    zip(source.parameters(), target.parameters()):
                target_param.data.copy_(source.param.data)

    def select_action(self, state, action_noise = None):
        actions = torch.zeros(self.num_agents, self.action_space)
        for agent in range(self.num_agents):
            state = state[agent, :].detach()
            action = self.policies[agent](state.unsqueeze(0)).squeeze()
            action += torch.from_numpy(np.random.randn(2) *
                    self.var[agent])
            if self.episode_counts > 1000 and self.var[agent] > 0.05:
                self.var[agent] *= 0.01
            action = torch.clamp(action, -1.0, 1.0)
            actions[agent, :] = action
        return actions

    def train(self, replay_buffer):
        if self.episode_counts <= 1000:
            return None, None
        policy_losses = []
        critic_losses = []
        for agent in range(self.num_agents):
            dynamics = replay_buffer.sample()
            batch = Dynamics(*zip(*dynamics))
            final_mask = torch.tensor(list(map(lambda state : state is not None,
                batch.next_states)), dtype = torch.uint8)
            states = torch.stack(batch.states)
            actions = torch.stack(batch.actions)
            rewards = torch.stack(batch.rewards)
            final_next_states = torch.stack([state
                for state in batch.next_states if state is not None])
            state = states.view(self.batch_size, -1)
            action = actions.view(self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            Q_current = self.critics[agent](state, action)
            final_next_actions = [self.policies_target[i](
                final_next_states[:, i, :])
                for i in range(self.num_agents)]
            final_next_actions = torch.stack(final_next_actions)
            final_next_actions = final_next_actions.transpose(0, 1).contiguous()
            Q_target = torch.zeros(self.batch_size)

            Q_target[fina_mask] = self.critics_target[agent](
                    final_next_states.view(-1,
                        self.num_agents * self.state_space),
                    final_next_actions.view(-1,
                        self.num_agents * self.action_space)).squeeze()
            Q_target = Q_target * self.gamma + \
                    (rewards[:, agent].unsqueeze(1) * self.epsilon)
            Q_loss = F.mse_loss(Q_current, Q_target)
            Q_loss.backward()
            self.critic_optimizer[agent].step()
            self.policy_optimizers[agent].zero_grad()
            agent_state = state[:, agent, :]
            agent_action = self.policies[agent](atent_state)
            actions[:, agent, :] = agent_action
            action = actions.view(self.batch_size, -1)
            policy_loss = -self.critics[agent](state, action).mean()
            policy_loss.backward()
            policy_optimizers[agent].step()
            policy_losses.append(policy_loss)
            critic_losses.append(critic_loss)
        if self.step_counts % 100 == 0 and self.step_counts > 0:
            for agent in range(self.num_agents):
                update_parameters(self.policies[agent],
                        self.policies_target[agent])
                update_parameters(self.ciritcs[agent],
                        self.critics_target[agent])
        return policy_losses, critic_losses
