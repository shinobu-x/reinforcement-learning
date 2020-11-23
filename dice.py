import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from distributions.categorical import Categorical
from modules.gru import GRU
from utils.compute_regularization import compute_regularization

# DiCE: The Infinitely Differentiable Monte-Carlo Estimator
# https://arxiv.org/abs/1802.05098
class Actor(nn.Module):
    def __init__(self, state_space, hidden_size):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_space, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, state):
        return  F.relu(self.l2(F.relu(self.l1(state))))

class Critic(nn.Module):
    def __init__(self, state_space, hidden_size):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_space, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

    def forward(self, stetes):
        x = F.relu(self.l2(F.relu(self.l1(state))))
        return self.l3(x)

class Discriminator(nn.Module):
    def __init__(self, state_space, action_space):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(state_space + action_space, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim = -1)
        return self.l3(F.relu(self.l2(F.relu(self.l1(x)))))

class BaseNN(nn.Module):
    def __init__(self, enable_rnn, rnn_num_inputs, hidden_size):
        super(BaseNN, self).__init__()
        self.hidden_size = hidden_size
        self.enable_rnn = enable_rnn
        if enable_rnn:
            self.gru = nn.GRU(rnn_num_inputs, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name: nn.init.constant_(param, 0)
                elif 'weight' in name: nn.init.orthogonal_(param)

    def gru_forward(self, state, hxs, masks):
        if state.size(0) == hxs.size(0):
            x, hxs = self.gru(state.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            n = hxs.size(0)
            t = int(x.size(0) / n)
            x = x.view(t, n, x.size(1))
            masks = masks.view(t, n)
            is_zero = ((mask[1:] == 0.0) \
                    .any(dim = -1).nonzero().squeeze().cpu())
            if is_zero.dim() == 0: is_zero = [is_zero.item() + 1]
            else: is_zero = (is_zero + 1).numpy().tolist()
            is_zero = [0] + is_zero + [t]
            hxs = hxs.unsqueeze(0)
            output = []
            for i in range(len(is_zero) - 1):
                start = is_zero[i]
                end = is_zero[i + 1]
                scores, hxs = self.gru(x[start: end],
                        hxs * masks[start].view(1, -1, 1))
                outputs.append(scores)

            x = torch.cat(outputs, dim = 0)
            x = x.view(t * n, -1)
            hxs = hxs.squeeze(0)
        return x, hxs

class BaseModel(BaseNN):
    def __init__(self, state_space, enable_rnn = False, hidden_size = 64):
        super(BaseModel, self).__init__(state_space, enable_rnn, hidden_size)
        if enable_rnn: state_space = hidden_size
        self.actor = Actor(state_space, hidden_size)
        self.critic = Critic(state_space, hidden_size)

    def forward(self, state, hxs, masks):
        if self.enable_rnn:
            state, hxs = self.gru_forward(state, hxs, masks)
        hx_actor = self.actor(state)
        value = self.critic(state)
        return value, actor_hx, hxs

class Agent(nn.Module):
    def __init__(self, state_space, action_space, enable_rnn = False):
        super(Agent, self).__init__()
        self.base_model = BaseModel(state_space, enable_rnn = True)
        #self.base_model = base_model(state_space, enable_rnn = True)
        num_outputs = action_space
        self.dist = Categorical(self.base_model.hidden_size, num_outputs)

    def select_action(self, states, hxs, masks, deterministic = False,
            reparam = False):
        value, actor_hx, hxs = self.base_model(states, hxs, masks)
        dist = self.dist(actor_hx)
        if deterministic: action = dist.mode()
        elif reparam: action = dist.rsample()
        else: action = dist.sample()
        log_probabilities = dist.log_probs(action)
        return value, action, log_probabilities, hxs

    def get_value(self, states, hxs, masks, action):
        value, actor_hx, hxs = self.base_model(states, hxs, masks)
        dist = self.dist(actor_hx)
        log_probabilities = dist.log_probs(action)
        entropy = dist.entropy().mean()
        return value, log_probabilities, entropy, hxs

class DICE(object):
    def __init__(self, env):
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.agent = Policy(self.state_space, self.action_space,
                enable_rnn = True)
        discriminator = Discriminator(self.state_space, self.action_space)
        self.clipping = 0.2
        self.regularization_type = 'KL'
        self.value_loss_coefficient = 0.5
        self.entropy_coefficient = 1e-2
        self.gamma = 0.99
        self.lambda = 0.9
        self.lr = 0.9
        self.epsilon = 0.01
        self.max_grad_norm = None
        self.use_clipped_value_loss = False
        self.enable_orthogonal_normalization = True
        self.num_discriminator_trains = 1
        self.discriminator_lr = 0.9
        self.policy_optimizer = Adam(policy.parameters(), lr = self.lr,
                eps = self.epsilon)
        self.optimizer_discriminator = Adam(discriminator.parameters(),
                lr = self.discriminator_lr * self.lr, eps = self.epsilon)
