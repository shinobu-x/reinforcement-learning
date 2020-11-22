import torch

class WeightRegularization:
    def __init__(self, regularization_type, regularization_factor = 1e-3,
            device = 'cpu'):
        self.regularization_type = regularization_type
        self.regularization_factor = regularization_factor
        self.device = device
        self.loss = torch.zeros(1).to(device)

    def compute_regularization(self, model):
        with torch.enable_grad():
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    if self.regularization_type == 'l1':
                        self.loss = self.loss + (self.regularization_factor *
                                torch.sum(torch.abs(param)))
                    elif self.regularization_type == 'l2':
                        self.loss = self.loss + (self.regularization_factor *
                                0.5 * torch.sum(torch.pow(param, 2)))
                    elif self.regularization_type == 'orthogonal':
                        param = param.view(param.shape[0], -1)
                        symbol = torch.mm(param, torch.t(param))
                        symbol -= torch.eye(param.shape[0])
                        self.loss = self.loss + (self.regularization_factor *
                                symbol.abs().sum())
        return self.loss
