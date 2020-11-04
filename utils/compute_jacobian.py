import torch
from compute_gradient import compute_gradient

def compute_jacobian(outputs, inputs, create_graph = True):
    outputs = [outputs] if torch.is_tensor(outputs) else list(outputs)
    inputs = [inputs] if torch.is_tensor(inputs) else list(inputs)
    jacobian = []
    for output in outputs:
        grad = torch.zeros_like(output.view(-1))
        for i in range(len(output.view(-1))):
            grad[i] = 1
            jacobian += [compute_gradient(output.view(-1), inputs, grad,
                retain_graph = True, create_graph = create_graph)]
            grad[i] = 0
    return torch.stack(jacobian)
