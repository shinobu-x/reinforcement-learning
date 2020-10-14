import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from utils import (p2v, v2p, flatten, compute_discounted_reward,
                   explained_variance_1d)

# Trust Region Policy Optimization
# https://arxiv.org/abs/1502.05477
class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(state_space, 16, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2)
        self.l1 = nn.Linear(2592, 256)
        self.l2 = nn.Linear(256, action_space)

    def forward(self, state):
        a = F.relu(self.conv1(state))
        a = F.relu(self.conv2(a))
        a = F.relu(self.l1(a.view(a.size(0), -1)))
        a = F.softmax(self.l2(a))
        return a

class Critic(nn.Module):
    def __init__(self, state_space):
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2)
        self.l1 = nn.Linear(2592, 256)
        self.l2 = nn.Linear(256, 1)

    def forward(self, state):
        out = F.relu(self.conv1(state))
        out = F.relu(self.conv2(out))
        out = F.relu(self.l1(out.view(out.size(0), -1)))
        out = self.l2(out)
        return out

class CriticExtension(nn.Module):
    def __init__(self, model, lr):
        super(CriticExtension, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.model = model
        self.lr = lr

    def fit(self, states, targets):
        def closure():
            predicts = self.predict(states)
            loss = F.mse_loss(predicts, targets)
            self.optimizer.zero_grad()
            loss.backward()
            return loss
        params_old = p2v(self.model.parameters())
        for lr in self.lr * 0.5 ** np.arange(10):
            self.optimizer = optim.LBFGS(self.model.parameters(), lr = lr)
            self.optimizer.step(closure)
            params_current = p2v(self.model.parameters())
            if any(np.isnan(params_current.data.cpu().numpy())):
                v2p(params_old, self.model.parameters())
            else:
                return

    def predict(self, states):
        return self.model.forward(torch.cat([torch.autograd.Variable(
            torch.Tensor(state).to(self.device)).unsqueeze(0) for state \
            in states]))

class TRPO(object):
    def __init__(self, env, num_episodes = 100, epsilon = 0.09, eta = 1e-3,
            gamma = 0.99):
        '''
        num_episodes: A number of episodes
        epsilon: KL constraint
        gamma: Discount factor
        eta: Penalty of the policy
        '''
        self.env = env
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.device = torch.device('cuda' if torch.cuda.is_available()
                else 'cpu')
        self.actor = Actor(self.state_space, self.action_space).to(self.device)
        self.critic = CriticExtension(Critic(self.state_space), lr = 0.1
                ).to(self.device)
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.eta = eta
        self.gamma = gamma

    def select_action(self, state):
        state = torch.Tensor(state).unsqueeze(0).to(self.device)
        probabilities = self.actor(torch.autograd.Variable(state,
            requires_grad = True))
        action = probabilities.multinomial(1)
        return action, probabilities

    def sample_trajectories(self):
        trajectories = []
        episode_current = 0
        entropy = 0
        while episode_current < self.num_episodes:
            episode_current += 1
            states, actions, rewards, action_probabilities = [], [], [], []
            state = self.env.reset()
            while True:
                states.append(state)
                action, probabilities = self.select_action(state)
                actions.append(action)
                action_probabilities.append(probabilities)
                entropy += -(probabilities * probabilities.log()).sum()
                state, reward, done, _ = self.env.step(action.data[0, 0])
                rewards.append(reward)
                if done:
                    trajectory = {'states': states,
                            'actions': actions,
                            'rewards': rewards,
                            'action_probabilities': action_probabilities}
                    trajectories.append(trajectory)
                    break
        states = flatten([trajectory['states'] for trajectory in trajectories])
        discounted_rewards = np.asarray(flatten([compute_discounted_reward(
            trajectory['rewards'], self.gamma) for trajectory in trajectories]))
        total_reward = sum(flatten([trajectory['rewards'] for trajectory
            in trajectories])) / self.num_episodes
        actions = flatten([trajectory["actions"] for trajectory
            in trajectories])
        action_probabilities = flatten([trajectory['action_probabilities']
            for trajectory in trajectories])
        entropy /= len(actions)
        return (states, discounted_rewards, total_reward, actions,
                action_probabilities, entropy)

    def compute_average_kl_divergence(self, model):
        states = torch.cat([torch.autograd.Variable(torch.Tensor(state).to(
            self.device)).unsqueeze(0) for state in self.states])
        policy_new = model(states) #.detach() + 1e-8
        policy_old = self.actor(states)
        return torch.sum(policy_old *
                torch.log(policy_old / policy_new), 1).mean()

    def hessian_vector_product(self, gradient):
        self.actor.zero_grad()
        average_kl_divergence = self.compute_average_kl_divergence(self.actor)
        kl_gradient = torch.autograd.grad(average_kl_divergence,
                self.actor.parameters(), create_graph = True)
        kl_gradient = torch.cat([grad.view(-1) for grad in kl_gradient])
        kl_gradient = torch.sum(kl_gradient * gradient)
        second_order_gradient = torch.autograd.grad(kl_gradient,
                self.actor.parameters())
        fisher_product = torch.cat([grad.contiguous().view(-1)
            for grad in second_order_gradient]).data
        return fisher_product + (self.eta * gradient.data)

    def conjugate_gradient(self, gradient):
        p = gradient.clone().data
        r = gradient.clone().data
        x = np.zeros_like(gradient.data.cpu().numpy())
        r2 = r.double().dot(r.double())
        for _ in range(10): # Run conjugate gradient algorithm
            z = self.hessian_vector_product(torch.autograd.Variable(p)
                    ).squeeze(0)
            v = r2 / p.double().dot(z.double())
            x += v.cpu().numpy() * p.cpu().numpy()
            r -= v * z
            r3 = r.double().dot(r.double())
            mu = r3 / r2
            p = r + mu * p
            r2 = r3
            if r2 < 1e-10: # Resudual tolerance
                break
        return x

    def surrogate_loss(self, theta):
        policy_target = deepcopy(self.actor)
        v2p(theta, policy_target.parameters())
        states = torch.cat([torch.autograd.Variable(torch.Tensor(state).to(
            self.device)).unsqueeze(0) for state in self.states])
        actions = torch.cat(self.actions)
        probability_current = self.actor(states).gather(1, actions).data + 1e-8
        probability_target = policy_target(states).gather(1, actions).data
        return -torch.mean((probability_target / probability_current) *
                self.advantage)

    def linesearch(self, gradient, full_steps, expected_improve_rate):
        accept_ratio = 0.1
        max_backtracks = 10
        f_value = self.surrogate_loss(gradient)
        num_searches = 0.5 ** np.arange(max_backtracks)
        for (_n_backtracks, step_frac) in enumerate(num_searches):
            gradient_new = gradient.data.cpu().numpy() + step_frac * full_steps
            f_value_new = self.surrogate_loss(torch.autograd.Variable(
                torch.from_numpy(gradient_new)))
            actual_improve = f_value - f_value_new
            expected_improve = expected_improve_rate * step_frac
            improve_ratio = actual_improve / expected_improve
            if improve_ratio > accept_ratio and actual_improve > 0:
                return torch.autograd.Variable(torch.from_numpy(gradient_new))
        return gradient

    def step(self, batch_size):
        (states, discounted_rewards, total_reward, actions,
                action_probabilities, self.entropy) = self.sample_trajectories()
        num_batches = len(actions) / batch_size \
                if len(actions) % batch_size == 0 \
                else (len(actions) / batch_size) + 1
        for batch in range(int(num_batches)):
            self.states = states[batch * batch_size:(batch + 1) * batch_size]
            self.discounted_rewards = discounted_rewards[batch * batch_size:(
                batch + 1) * batch_size]
            self.actions = actions[batch * batch_size:(batch + 1) * batch_size]
            self.action_probabilities = action_probabilities[batch *
                    batch_size:(batch + 1) * batch_size]
            # Compute the normalized advantage
            baseline = self.critic.predict(self.states).data
            discounted_rewards = torch.Tensor(self.discounted_rewards
                    ).unsqueeze(1).to(self.device)
            advantage = discounted_rewards - baseline
            self.advantage = (advantage - advantage.mean()) / \
                    (advantage.std() + 1e-8)
            # Compute the probability ration of taken actions
            policy_target = torch.cat(self.action_probabilities).gather(1,
                    torch.cat(self.actions))
            policy_current = policy_target.detach() + 1e-8
            probability_ratio = policy_target / policy_current
            # Compute the surrogate loss
            surrogate_loss = -torch.mean(probability_ratio * \
                    torch.autograd.Variable(self.advantage)) - \
                    (self.eta * self.entropy)
            # Compute the gradient of the surrogate loss
            self.actor.zero_grad()
            surrogate_loss.backward(retain_graph = True)
            # Parameter to vector
            vector = []
            [vector.append(param.view(-1)) for param in
                    self.actor.parameters()]
            vector = torch.cat(vector)
            policy_gradient = vector.squeeze(0)
            if policy_gradient.nonzero().size()[0]:
                # Conjugate gradient to determine the step direction
                step_direction = self.conjugate_gradient(-policy_gradient)
                step_direction_variable = torch.autograd.Variable(
                        torch.from_numpy(step_direction))
                # Linesearch to determine the step size
                hessian = self.hessian_vector_product(step_direction_variable
                    ).cpu().numpy().T
                lm = np.sqrt(0.5 * step_direction.dot(hessian) / self.epsilon)
                full_step = step_direction / lm
                gradient_dot_step_direction = -policy_gradient.dot(
                        step_direction_variable).item()#.data[0]
                theta = self.linesearch(p2v(self.actor.parameters()),
                        full_step, gradient_dot_step_direction / lm)
                # Fit the estimated value to the observed discounted rewards
                if self.discounted_rewards.ndim == 1:
                    evaluation_before = explained_variance_1d(
                            baseline.squeeze(1).cpu().numpy(),
                            self.discounted_rewards)
                    self.critic.zero_grad()
                    critic_parameters = p2v(self.critic.parameters())
                    self.critic.fit(self.states, torch.autograd.Variable(
                        torch.Tensor(self.discounted_rewards).to(self.device)))
                    evaluation_after = explained_variance_1d(
                            self.critic.predict(
                                self.states).data.squeeze(1).cpu().numpy(),
                            self.discounted_rewards)
                if evaluation_after < evaluation_before or \
                        np.abs(evaluation_after) < 1e-4:
                            v2p(critic_parameters, self.critic.parameters())
                # Update policy parameters
                policy_current = deepcopy(self.actor)
                policy_current.load_state_dict(self.actor.state_dict())
                if not any(np.isnan(theta.data.cpu().numpy())):
                    v2p(theta, self.actor.parameters())
                average_kl_divergence = self.compute_average_kl_divergence(
                        policy_current)
                diagnostics = OrderedDict([
                    ('Total reward', total_reward),
                    ('KL divergence', average_kl_divergence.item()),
                    ('Entropy', self.entropy.item()),
                    ('Evaluation before', evaluation_before),
                    ('Evaluation after', evaluation_after)])
                for k, v in diagnostics.items():
                    print('{}: {}'.format(k, v))
                return total_reward
