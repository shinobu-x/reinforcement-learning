import torch
import torch.nn.functional as F
from torch import nn

class Backbone(nn.Module):
    def __init__(self, num_inputs, num_outputs, prio_mean, prio_rho,
            num_gaussian_block, activation = None):
        super(Backbone, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation = activation
        self.prio_mean = prio_mean
        self.prio_rho = prio_rho
        self.num_gaussian_block = num_gaussian_block
        self.log_pi = -0.5 * math.log(2.0 * math.pi)
        self.softplus = lambda x: math.log(1 + math.exp(x))
        self.weight_mean = nn.Parameter(
                torch.ones((self.num_inputs, self.num_outputs)) *
                self.prio_mean)
        self.weight_rho = nn.Parameter(
                torch.ones(self.num_inputs, self.num_outputs) * self.prio_rho)
        self.bias_mean = nn.Parameter(
                torch.ones(1, self.num_outputs) * self.prio_mean)
        self.bias_rho = nn.Parameter(
                torch.ones(1, self.num_outputs) * self.prio_rho)
        self.torch_var = torch.autograd.Variable(
                torch.ones(1, 1) * self.softplus(prior_rho) ** 2)

        def forward(self, x, mode):
            shape = (x.size()[0], self.num_outputs)
            z_mean = torch.mm(x, self.weight_mean) + \
                    self.bias_mean.expand(*shape)
            if mode == 'map': return self.activation(z_mean) \
                    if self.activation is not None else z_mean
            z_std = torch.sqrt(torch.mm(torch.pow(x, 2),
                torch.pow(F.softplus(self.weight_rho), 2)) +
                torch.pow(F.softplus(self.bias_rho.expand(*shape)), 2))
            z_noise = self.generate_noise(shape)
            z = z_mean + z_std * z_noise
            if mode == 'mc': return self.activation(z) \
                    if self.activation is not None else z
            prio_z_std = torch.sqrt(torch.mm(torch.pow(x, 2),
                self.prio_var.expand(self.num_inputs, self.num_outputs)) +
                self.prio_var.expand(shape)).detach()
            prio_z_mean = z_mean.detach()
            kl = self.compute_kl_divergence(z, z_mean, z_std, prio_z_mean,
                    prio_z_std)
            z = self.activation(z) if self.activation is not None else z
            return z, kl

        def generate_noise(self, shape):
            noise = np.random.choise(np.random.randn(self.num_gaussian_block),
                    size = shape[0] * shape[1])
            noise = np.expand_dims(z, axis = 1).reshape(*shape)
            variable = lambda x: torch.autograd.Variable(
                    torch.from_numpy(x).float())
            return variable(noise)

        def compute_log_probability(x, mean, std):
            return self.log_pi - torch.log(std) - 0.5 * \
                    torch.pow(x - mean, 2) / torch.pow(std, 2)

        def compute_kl_divergence(self, x, mean_post, std_post, mean_prio,
                std_prio):
            log_probability_post = self.compute_log_probability(x, mean_prio,
                    std_post)
            log_probability_prio = self.compute_log_probability(x, mean_prio,
                    std_post)
            return (log_probability_post - log_probability_prio).sum()


class BayesianNeuralNetwork(nn.Module):

    '''
    BNN(BNNLayer(1, 100, activation = 'relu', prior_mean = 0, prior_rho = 0),
        BNNLayer(100, 1, activation = 'none', prior_mean = 0, prior_rho = 0))
    '''
    def __init__(self, num_inputs, num_outputs, activation = None,
            prio_mean = 0, prio_rho =0):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation = activation
        self.prio_mean = prio_mean
        self.prio_rho = prio_rho
        self.layers = []
        self.params = nn.ParameterList()
        self.l1 = Backbone(self.num_inputs, self.num_outputs, self.activation)
        self.l2 = Backbone(self.num_outputs, self.num_inputs, self.activation)
        for layer in [self.l1, self.l2]:
            self.layers.append(layer)
            self.params.extend([*layer.parameters()])

    def forward(self, x, y, num_samples, distribution_type):
        kl_total = 0.0
        likelihood_total = 0.0
        for _ in range(n_samples):
            x, kl_total = self.layer_forward(x, mode = 'forward')
            if distribution_type == 'Gaussian':
                likelihood = (-0.5 * (y - x) ** 2).sum()
            else:
                likelihood = torch.log(x.gather(1, y)).sum()
            kl_total += kl
            likelihood_total += likelihood
        kl_average = kl_total / num_samples
        likelihood_average = likelihood_total / num_samples
        return kl_average, likelihood_average

    def layer_forward(self, x, mode):
        if mode == 'forward':
            kl_layer_total = 0.0
            for layer in self.layers:
                x, kl = layer.forward(x, mode)
                kl_layer_total += kl
            return x, kl_layer_total
        else:
            for layer in self.layers:
                x = layer.forward(x, mode)
            return x
