import torch
from compute_gradient import compute_gradient

def compute_hessian(outputs, inputs, out = None, allow_unused = False,
        retain_graph = False, create_graph = False):
    inputs = [inputs] if torch.is_tensor(inputs) else list(inputs)
    numel = sum(input.numel() for input in inputs)
    out = outputs.new_zeros(numel, numel) if out is None else out
    row_index = 0
    for i, input in enumerate(inputs):
        [grad] = torch.autograd.grad(outputs, input,
                create_graph = create_graph, allow_unused = allow_unused)
        grad = torch.zeros_like(input) if grad is None else grad
        grad = grad.contiguous().view(-1)
        for j in range(input.numel()):
            row = compute_gradient(grad[j], inputs[i:],
                    retain_graph = retain_graph,
                    create_graph = create_graph)[j:] \
                                if grad[i].requires_grad \
                                else grad[j].new_zeros(sum(x.numel() \
                                for x in inputs[i:]) -j)
            out[row_index, row_index:].add_(row.type_as(out))
            if row_index + 1 < numel:
                out[row_index + 1:, row_index].add_(row[1:].type_as(out))
            del row
            row_index += 1
        del grad
    return out
