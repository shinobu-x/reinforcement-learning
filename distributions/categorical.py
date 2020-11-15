from torch import nn
from torch.distributions import Categorical
from torch.nn import init

categorical = Categorical
categorical.mode = lambda self: self.probs.argmax(dim = -1, keepdim = True)
sample = categorical.sample
log_prob = categorical.log_prob
categorical.log_probs = lambda self, actions: log_prob(
        self, actions.squeeze(-1)).view(
                actions.size(0), -1).sum(-1).unsqueeze(-1)

class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()
        self.l = nn.Linear(num_inputs, num_outputs)
        init.orthogonal_(self.l.weight.data, gain = 1e-2)
        init.constant_(self.l.bias.data, 0)

    def forward(self, x):
        return categorical(logits = self.l(x))
