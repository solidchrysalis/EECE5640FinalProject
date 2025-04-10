import torch
import stochastic_cuda

w = torch.ones(1000, device='cuda', requires_grad=True)
target = torch.zeros_like(w)
loss_fn = torch.nn.MSELoss()

for i in range(100):
    w.grad = None
    loss = loss_fn(w, target)
    loss.backward()
    stochastic_cuda.stochastic(w.data, w.grad, lr=0.1)
    print(f"Step {i}, Loss: {loss.item()}")
