import torch
import adagrad_cuda
import time

w = torch.ones(1000, device='cuda', requires_grad=True)
target = torch.zeros_like(w)
loss_fn = torch.nn.MSELoss()

for i in range(100):
    w.grad = None
    loss = loss_fn(w, target)
    loss.backward()
    start = time.monotonic_ns()
    adagrad_cuda.adagrad(w.data, w.grad, 0.1)
    print(f"Step {i}, Loss: {loss.item()}")
    stop = time.monotonic_ns()
    print(f"Time: {(stop - start) * 1000}")

