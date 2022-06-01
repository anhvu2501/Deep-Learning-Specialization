# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop:
#   - forward pass: compute predition
#   - backward pass: gradients
#   - update weights
import torch

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


# model prediction
def forward(x):
    return w * x


# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()


print(f'Predition before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # Prediction = forward pass
    y_pred = forward(X)

    # Loss
    l = loss(Y, y_pred)

    # Gradients = backward pass
    l.backward()

    # Update weights
    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero gradients
    w.grad.zero_()

    if epoch % 10 == 0:
        print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {forward(5):.3f}')
