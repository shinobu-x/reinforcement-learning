import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

#Attention-based Curiosity-driven Exploration in Deep Reinforcement Learning
# https://arxiv.org/abs/1910.10840
class ConvolutionNetwork(nn.Module):
    def __init__(self, num_stacked_frames = 4):
        super.__init__()
        self.num_stacked_frames = num_stacked_frames
        self.out_channels = 32
        self.kernel_size = 3
        self.stride = 2
        self.padding = self.kernel_size // 2
        self.conv1 = nn.Conv2d(self.num_stacked_frames, self.out_channels,
                self.kernel_size, self.stride, self.padding)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels,
                self.kernel_size, self.stride, self.padding)
        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels,
                self.kernel_size, self.stride, self.padding)
        self.conv4 = nn.Conv2d(self.out_channels, self.out_channels,
                self.kernel_size, self.stride, self.padding)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = nn.AvgPool2d(2)(x)
        return x.view(x.size(0), -1)

class AttentionNetwork(nn.Module):
    def __init__(self, num_features):
        super.__init__()
        self.num_features = num_features
        self.attn = nn.Linear(self.num_features, self.num_features)

    def forward(self, target, attention = None):
        return target * F.softmax(self.attn(target if attention is None
            else attention), dim = -1)

class EncoderNetwork(nn.Module):
    def __init__(self, num_stacked_frames, num_features, hidden_size,
            enable_lstm = True):
        super.__init__()
        self.num_stacked_frames = num_stacked_frames
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.enable_lstm = enable_lstm
        self.conv = ConvolutionNetwork(
                num_stacked_frames = self.num_stacked_frames)
        if self.enable_lstm:
            self.lstm = nn.LSTMCell(self.num_features, self.hidden_size)

    def reset_internal_state(self, num_buffers = None, indices = None):
        if self.enable_lstm:
            with torch.no_grad():
                if indices is None:
                    self.hidden_target = torch.zeros(
                            num_buffers, self.hidden_size,
                            device = self.lstm.weight_ih.device)
                    self.current_target = torch.zeros(
                            num_buffers, self.hidden_size,
                            device = self.lstm.weight_ih.device)
                else:
                    indices = torch.as_tensor(indices.astype(np.uint8),
                            device = self.lstm.weight_ih.device)
                    if indices.sum():
                        self.hidden_target = (
                                1 - indices.view(-1, 1)).float() * \
                                        self.hidden_target
                        self.current_target = (
                                1 - indices.view(-1, 1)).float() * \
                                        self.current_target

    def forward(self, x):
        x = self.conv(x)
        if self.enable_lstm:
            x = x.view(-1, self.num_stacked_frames)
            self.hidden_target = self.lstm(
                    x, (self.hidden_target, self.current_target))
            self.current_target = self.lstm(
                    x, (self.hidden_target, self.current_target))
            return self.hidden_target, self.current_target
        else:
            return x.view(-1, self.num_stacked_frames)

class ForwardNetwork(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = 256
        self.num_outputs = 288
        self.l1 = nn.Linear(self.num_features, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.num_outputs)

    def forward(self, x):
        return self.l2(self.l1(x))

class BackwardNetwork(nn.Module):
    def __init__(self, action_space, num_features = 288):
        self.action_space = action_space
        self.num_features = num_features
        self.hidden_size = 256
        self.l1 = nn.Linear(self.num_features * 2, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.action_space)

    def forward(self, x):
        return self.l2(self.l1(x))

class AdversarialNetwork(nn.Module):
    def __init__(self, num_features, action_space, attention_type = 'single'):
        super().__init__()
        self.num_features = num_features
        self.action_space = action_space
        self.attention_type = attention_type
        self.forward_network = ForwardNetwork(self.num_features +
                self.action_space)
        self.backward_network = BackwardNetwork(self.action_space,
                self.num_features)
        if self.attention_type == 'single':
            self.forward_attention = AttentionNetwork(self.num_features +
                    self.action_space)
            self.backward_attention = AttentionNetwork(2 * self.num_features)
        else:
            self.forward_feature_attention = AttentionNetwork(self.num_features)
            self.forward_action_attention = AttentionNetwork(self.action_space)
            self.backward_current_feature_attention = AttentionNetwork(
                    self.num_features)
            self.backward_next_feature_attention = AttentionNetwork(
                    self.num_features)

    def forward(self, current_feature, next_feature, current_action):
        one_hot_encode = torch.zeros(
                current_action.shape[0], action_space,
                device = forward_network.l1.weight.device).scatter_(
                1, current_action.long().view(-1, 1), 1)
        forward_attention_in = self.forward_attention(
                 torch.cat((current_feature, one_hot_encode), 1)) \
                         if self.attention_type == 'single' else \
                         torch.cat((
                             self.forward_feature_attention(current_feature),
                             self.forward_action_attention(one_hot_encode)), 1)
        next_feature_prediction = self.forward_network(forward_attention_in)
        backward_attention_in = self.backward_attention(torch.cat((
            current_feature), self.backward_next_feature_attention(
                next_feature)), 1) if self.attention_type == 'single' \
                        else torch.cat((self.backward_current_feature_attention(
                            current_feature),
                            self.backward_next_feature_attenton(next_feature)),
                            1)
        action_prediction = self.backward_network(backward_attention_in)
        return next_feature_prediction, action_prediction

class IntrinsicCuriosityNetwork(nn.Module):
    def __init__(self, num_stacked_frames, action_space, attention_type,
            num_inputs = 288, num_features = 256):
        super().__init__()
        self.num_stacked_frames = num_stacked_frames
        self.action_space = action_space
        self.attention_type = attention_type
        self.num_inputs = num_inputs
        self.num_features = num_features
        self.encoder_network = EncoderNetwork(self.num_stacked_frames,
                self.num_inputs, enable_lstm = True)
        self.adversarial_network = AdversarialNetwork(self.num_inputs,
                self.action_space, self.attention_type)

    def forward(self, num_envs, states, actions):
        encoded_features = self.encoder_network(states)
        features = encoded_features[0: -num_envs]
        next_features = encoded_features[num_envs:]
        next_feature_prediction, action_prediction = self.adversarial_network(
                features, next_features, actions)
        loss = self.compute_loss(next_feature, next_feature_predition, actions,
                action_prediction)
        return loss

    def compute_loss(self, next_features, next_feature_prediction, actions,
            action_prediction):
        if self.attention_type != 'single':
            forward_loss = F.mse_loss(next_features, next_feature_prediction)
        else:
            forward_attention = AttentionNetwork(self.num_features)
            forward_loss = forward_attention(F.mse_loss(
                next_features, next_feature_prediction, reduction = 'none'),
                next_features).mean()
        backward_loss = F.cross_entropy(action_prediction.view(-1,
            self.action_space), actions.long())
        return forward_loss, backward_loss

class Actor(nn.Module):
    def __init__(self, num_stacked_frames, action_space, encoder_network,
            num_inputs = 288):
        super().__init__()
        self.num_stacked_frames = num_stacked_frames
        self.num_inputs = num_inputs
        self.action_space = action_space
        self.encoder_network = encoder_network
        self.attention_network = AttentionNetwork(
                self.encoder_network.hidden_size)
        self.l = nn.Linear(self.encoder_network.hidden_size, self.action_space)

    def forward(self, states):
        features = self.encoder_network(states)
        probability = self.l(self.attention_network(features))
        return probability, features

class Critic(nn.Module):
    def __init__(self, num_stacked_frames, encoder_network, num_inputs = 288):
        super().__init__()
        self.num_stacked_frames = num_stacked_frames
        self.num_inputs = num_inputs
        self.encoder_network = encoder_network
        self.attention_network = AttentionNetwork(
                self.encoder_network.hidden_size)
        self.l = nn.Linear(self.encoder_network.hidden_size, 1)

    def forward(self, states):
        features = self.encoder_network(states)
        value = self.l(self.attention_network(features)).squeeze()
        return value

class A2C(nn.Module):
    def __init__(self, num_stacked_frames, action_space, num_inputs = 288):
        super().__init__()
        self.num_stacked_frames = num_stacked_frames
        self.num_inputs = num_inputs
        self.action_space = action_space
        self.encoder_network = EncoderNetwork(self.num_stacked_frames,
                self.num_inputs)
        self.actor = Actor(self.num_stacked_frames, self.num_inputs,
                self.encoder_network)
        self.critic = Critic(self.num_stacked_frames, self.num_inputs,
                self.encoder_network)

    def forward(self, states):
        probability, features = self.actor(states)
        value = self.critic(states)
        return probability, value, features

    def select_action(self, states):
        probability, value, features = self.forward(states)
        probability = F.softmax(probability, dim = -1)
        distribution = Categorical(probability)
        action = distribution.sample()
        log_probability = distribution.log_prob(action)
        entropy_mean = distribution.entropy().mean()
        return action, log_probability, entropy_mean, value, features
