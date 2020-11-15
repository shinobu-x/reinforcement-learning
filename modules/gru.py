from torch import nn

class GRU(nn.Module):
    def __init__(self, state_space, hidden_size, enable_recurrent = True):
        super(GRU, self).__init__()
        self.enable_recurrent = enable_recurrent
        self.hidden_size = hidden_size
        if enable_recurrent:
            self.gru = nn.GRU(state_space, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name: nn.init.constant_(param, 0)
                elif 'weight' in name: nn.init.orthogonal_(param)

    def forward(self, states, hxs, masks):
        if inputs.size(0) == hxs.size(0):
            x, hxs = self.gru(inputs.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x, hxs = x.squeeze(0), hxs.squeeze(0)
        else:
            n = hxs.size(0)
            t = int(x.size(0) / n)
            x = inputs.view(t, n, inputs.size(1))
            masks = masks.view(t, n)
            is_zero = ((mask[1:] == 0.0)\
                    .any(dim = -1).nonzero().squeeze().cpu())
            if is_zero.dim() == 0: is_zero = [is_zero.item() + 1]
            else: is_zero = (is_zero + 1).numpy().tolist()
            is_zero = [0] + is_zero + [t]
            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(is_zero) - 1):
                start = is_zero[i]
                end = is_zero[i + 1]
                score, hxs = self.gru(x[start: end],
                        hxs * mask[start].view(1, -1, 1))
                outputs.append(score)
            x = torch.cat(outputs, dim = 0)
            x = x.view(t * n, -1)
            hxs = hxs.squeeze(0)
        return x, hxs
