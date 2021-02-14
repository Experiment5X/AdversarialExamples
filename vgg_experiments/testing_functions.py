import numpy as np
import torch


def cool_function(x):
    x_var = torch.Tensor([x])
    x_var.requires_grad = True
    output = 3 * x_var * x_var + 2
    output.backward()

    return output.data.numpy(), x_var.data.numpy(), x_var.grad


"""
Questions:
    - What is .data.numpy() vs .numpy()?
    - What does 'requires_grad=True' do?
        - Means that the gradient will be computed, it has a .grad_fn attribute. There may be more to it.
    - What is a Variable?
        - A deprecated API for using autograd
"""

print(cool_function(5))
