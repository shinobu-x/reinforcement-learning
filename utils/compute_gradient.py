import torch

def compute_gradient(outputs, inputs, grad_outputs = None, retain_graph = False,
        create_graph = False):
    outputs = [outputs] if torch.is_tensor(outputs) else list(outputs)
    inputs = [inputs] if torch.is_tensor(inputs) else list(inputs)
    grads = torch.autograd.grad(outputs, inputs, grad_outputs,
            allow_unused = True, retain_graph = retain_graph,
            create_graph = create_graph)
    grads = [grad if grad is not None else torch.zeros_like(input)
            for grad, input in zip(grads, inputs)]
    return torch.cat([grad.contiguous().view(-1) for grad in grads])
