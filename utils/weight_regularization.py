import torch

class WeightRegularization:
    def __init__(self, regularization_type, regularizer, device = 'cpu'):
        self.regularization_type = regularization_type
        self.regularizer = regularizer
        self.device = device
        self.loss = torch.zeros(1).to(device)

    def compute_regularization(self, model):
        with torch.enable_grad():
            for name, param in model.named_parameters():
                if 'bias' not in name:
                    if self.regularization_type == 'l1':
                        self.loss = self.loss + (self.regularizer *
                                torch.sum(torch.abs(param)))
                    elif self.regularization_type == 'l2':
                        self.loss = self.loss + (self.regularizer *
                                0.5 * torch.sum(torch.abs(param, 2)))
                    elif self.regularization_type == 'orthogonal':
                        param = param.view(param.shape[0], -1)
                        symbol = torch.mm(param, torch.t(param))
                        symbol -= torch.eye(param.shape[0])
                        self.loss = self.loss + (self.regularizer *
                                symbol.abs().sum())
        return loss
