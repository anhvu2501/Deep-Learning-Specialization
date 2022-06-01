import torch
import numpy as np

# Everything in pytorch is defines as tensor
x = torch.empty(1)  # tensor of 1 elemenet => scalar value
x = torch.empty(3)  # tensor of 3 elements => just like a vector of 3 elements
x = torch.empty(2, 3)  # 2d tensor
# ... and so on

x = torch.zeros(2, 2, 3, dtype=torch.float16)
y = torch.ones(2, 2, 2, dtype=torch.float32)

x = torch.tensor([2.5, 0.1])

# In pytorch, every function has a trailing underscore will do an in place operation that will modify
# the variable that it is applied on. Eg: add_, tensor_, etc.
x = torch.rand(2, 2)
y = torch.rand(2, 2)

z = x + y
z = torch.add(x, y)
y.add_(x)
print(z)
print(y)

# Reshape a tensor
x = torch.rand(4, 4)
print(x)
y = x.view(2, 8)
print(y)
y = x.view(16)
print(y)

# Convert from tensor to numpy array and revert
# Note: the memory location of tensor and numpy array (tensor can be in both CPU and GPU)
# but numpy array only in the CPU
# Ref: https://nttuan8.com/bai-1-tensor/#Torch_Tensors
x = torch.ones(5)
x_np = x.numpy()

x_np = np.array([1, 2, 3])
x_cpu = torch.from_numpy(x_np)

cuda = torch.cuda.is_available()
print(cuda)




