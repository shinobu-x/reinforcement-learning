import torch
from torch import nn
from torch.nn import functional as F

# Attention Augmented Convolutional Networks
# https://arxiv.org/abs/1904.09925
class AttentionAugmentedConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size,
            key_dimension, value_dimension, attention_maps, shape = 0,
            relative = False, stride = 1):
        super(AttentionAugmentedConv2d, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.key_dimension = key_dimension
        self.value_dimension = value_dimension
        self.attention_maps = attention_maps
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2
        self.device = torch.device('cuda' if torch.cuda.is_available() \
                else 'cpu')
        self.conv = nn.Conv2d(self.input_channels,
                self.output_channels - self.value_dimension, self.kernel_size,
                stride = self.stride, padding = self.padding)
        self.qkv = nn.Conv2d(self.input_channels,
                2 * self.key_dimension + self.value_dimension,
                self.kernel_size, stride = self.stride, padding = self.padding)
        self.att = nn.Conv2d(self.value_dimension, self.value_dimension,
                kernel_size = 1, stride = 1)
        if self.relative:
            self.key_relative_x = nn.Parameter(torch.randn((
                2 * self.shape - 1, key_dimension // attention_maps),
                requires_grad = True))
            self.key_relative_y = nn.Parameter(torch.randn((
                2 * self.shape - 1, key_dimension // attention_maps),
                requires_grad = True))

    def forward(self, x):
        conv_out = self.conv(x)
        batch_size, _, height, width = conv_out.size()
        flat_query, flat_key, flat_value, query, key, value = \
                self.compute_flat_qkv(x, self.key_dimension,
                        self.value_dimension, self.attention_maps)
        logits = torch.matmul(flat_query.transpose(2, 3), flat_key)
        if self.relative:
            relative_logits_y, relative_logits_x = \
                    self.compute_relative_logits(query)
            logits += relative_logits_y
            logits += relative_logits_x
        # O_h = Softmax((Q * K^T + S^rel_H + S^rel_W) / sqrt(d^h_k)) * V
        weights = F.softmax(logits, -1)
        att_out = torch.matmul(weights, flat_value.transpose(2, 3))
        att_out = torch.reshape(att_out, (batch_size, self.attention_maps,
            self.value_dimension // self.attention_maps, height, width))
        att_out = self.combine_heads(att_out)
        att_out = self.att(att_out)
        # AAConv(X) = Concat[Conv(X), MHA(X)]
        return torch.cat((conv_out, att_out), 1)

    def compute_flat_qkv(self, x, key_dimension, value_dimension,
            attention_maps):
        qkv = self.qkv(x)
        batch_size, _, height, width = qkv.size()
        query, key, value = torch.split(qkv, [key_dimension, key_dimension,
            value_dimension], 1)
        query = self.split_heads(query, attention_maps)
        key = self.split_heads(key, attention_maps)
        value = self.split_heads(value, attention_maps)
        key_head_dimension = key_dimension // attention_maps
        query *= key_head_dimension ** -0.5
        flat_query = torch.reshape(query, (batch_size, attention_maps,
            key_dimension // attention_maps, height * width))
        flat_key = torch.reshape(key, (batch_size, attention_maps,
            key_dimension // attention_maps, height * width))
        flat_value = torch.reshape(value, (batch_size, attention_maps,
            value_dimension // attention_maps, height * width))
        return flat_query, flat_key, flat_value, query, key, value

    def split_heads(self, x, attention_maps):
        batch_size, channels, height, width = x.size()
        shape = (batch_size, attention_maps, channels // attention_maps,
                height, width)
        x = torch.reshape(x, shape)
        return x

    def combine_heads(self, x):
        batch_size, attention_maps, value_dimension, height, width = x.size()
        shape = (batch_size, attention_maps * value_dimension, height, width)
        x = torch.reshape(x, shape)
        return x

    def compute_relative_logits(self, query):
        batch_size, attention_maps, key_dimension, height, width = query.size()
        query = torch.transpose(query, 2, 4).transpose(2, 3)
        relative_logits_x = self.compute_relative_logits_1d(query,
                self.key_relative_x, height, width, attention_maps, 'x')
        relative_logits_y = self.compute_relative_logits_1d(
                torch.transpose(query, 2, 3), self.key_relative_y, width,
                height, attention_maps, 'y')
        return relative_logits_x, relative_logits_y

    def compute_relative_logits_1d(self, query, key_relative, height, width,
            attention_maps, axis):
        relative_logits = torch.einsum('bhxyd,md->bhxym', query,
                key_relative)
        relative_logits = torch.reshape(relative_logits, (-1,
            attention_maps * height, width, 2 * width - 1))
        relative_logits = self.compute_abs(relative_logits)
        relative_logits = torch.reshape(relative_logits, (-1, attention_maps,
            height, width, width))
        relative_logits = torch.unsqueeze(relative_logits, 3)
        relative_logits = relative_logits.repeat((1, 1, 1, height, 1, 1))
        if axis == 'x':
            relative_logits = torch.transpose(relative_logits, 3, 4)
        else:
            reletive_logits = torch.transpose(relative_logits, 2, 4)
            relative_logits = torch.transpose(relative_logits, 4, 5)
            relative_logits = torch.transpose(relative_logits, 3, 5)
        relative_logits = torch.reshape(relative_logits, (-1, attention_maps,
            height * width, height * width))
        return relative_logits

    def compute_abs(self, x):
        batch_size, attention_maps, length, _ = x.size()
        col_pad = torch.zeros((batch_size, attention_maps, length, 1))
        x = torch.cat((x, col_pad), 3)
        flat_x = torch.reshape(x, (batch_size, attention_maps, length ** 2 * 2))
        flat_pad = torch.zeros((batch_size, attention_maps, length - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), 2)
        final_x = torch.reshape(flat_x_padded, (batch_size, attention_maps,
            length + 1, length * 2 - 1))
        final_x = final_x[:, :, :length, length - 1:]
        return final_x

