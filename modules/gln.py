import math
import numpy as np
import sys
import torch

class Node(object):
    def __init__(self, num_in_features, input_size, num_contexts, distribution):
        self.num_in_features = num_in_features
        self.num_contexts = num_contexts
        self.context_size = 2 ** num_contexts
        self.w = torch.zeros(self.context_size, num_in_features)
        self.context_vector = [distribution.sample([input_size]).view(-1)
                for _ in range(num_contexts)]
        self.context_bias = [0.0 for _ in range(num_contexts)]

    def get_context(self, x):
        r = 0
        for i in range(self.num_contexts):
            if torch.dot(x, self.context_vector[i]) >= self.context_bias[i]:
                r += 2 ** i
        return r

    def compute_logit(x, epsilon = 1e-6):
        return torch.log(x / (1 - x + epsilon) + epsilon)

    def clip(x, bound):
        x[x > bound] = bound
        x[x < -1 * bound] = -1 * bound
        return x

    def forward(self, p, z):
        context = self.get_context(z)
        return torch.sigmoid(torch.dot(self.w[context],
            Node.compute_logit(p))), p, context

    def backward(self, forward, target, p, context, lr, bound = 200):
        epsilon = 1e-6
        if target == 0: loss = -1 * torch.log(min(1 - forward + epsilon,
            torch.as_tensor(1 - epsilon)))
        else: loss = -1 * torch.log(min(forward + epsilon,
            torch.as_tensor(1 - epsilon)))
        if torch.isnan(loss):
            sys.exit()
        self.w[context] = Node.clip(self.w[context] - lr * (forward - target) *
                Node.compute_logit(p), bound)

class Layer(object):
    def __init__(self, num_in_features, num_nodes, input_size, num_contexts,
            distribution):
        self.num_in_features = num_in_features
        self.nodes = [Node(num_in_features + 1, input_size, num_contexts,
            distribution) for _ in range(num_nodes)]
        self.bias = math.e / (math.e + 1)

    def __call__(self, p, z):
        return self.forward(p, z)

    def forward(self, p, z):
        if p is not None: p = torch.cat((torch.as_tensor([self.bias]), p))
        else: p = torch.cat((torch.as_tensor([self.bias]),
            0.5 * torch.ones(self.num_in_features)))
        return [self.nodes[i].forward(p, z) for i in range(len(self.nodes))]

    def backward(self, forward, target, lr, bound = 200):
        loss = []
        for i in range(len(self.nodes)):
            loss.append(self.nodes[i].backward(forward[i][0], target,
                forward[i][1], forward[i][2], lr, bound))
        return loss

class GLN(object):
    def __init__(self, num_in_features, num_nodes, input_size, num_contexts,
            distribution):
        self.layers = [Layer(num_in_features, num_nodes[0], input_size,
            num_contexts, distribution)]
        self.layers = self.layers + [Layer(num_nodes[i - 1], num_nodes[i],
            input_size, num_contexts, distribution)
            for i in range(1, len(num_nodes))]

    def train(self, z, target, lr):
        z = z.view(-1)
        for i in range(len(self.layers)):
            if i == 0:
                forward = self.layers[i].forward(None, z)
                self.layers[i].backward(forward, target, lr)
            else:
                p = torch.cat([forward[i][0].unsqueeze(0)
                    for i in range(self.layers[i].num_in_features)])
                forward = self.layers[i].forward(p, z)
                loss = self.layers[i].backward(forward, target, lr)
        return loss

    def inference(self, z):
        z = z.view(-1)
        for i in range(len(self.layers)):
            if i == 0: forward = self.layers[i].forward(None, z)
            else:
                p = torch.cat([forward[i][0].unsqueeze(0)
                    for i in range(self.layers[i].num_in_features)])
                forward = self.layers[i].forward(p, z)
        return forward[0][0]
