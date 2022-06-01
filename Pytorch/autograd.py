# Gradient calculation

import torch

# If we want to calculate the gradients of some function later
# => must specify the argument requires_grad=True
x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2
z = y * y * 2
z = z.mean()
print(z)

z.backward()  # dz/dx
print(x.grad)

# If we want to work with the same tensor, but without requiring grad
x.requires_grad_(False)
print(x)

y = x.detach()
print(y)

with torch.no_grad():
    y = x
    print(y)
