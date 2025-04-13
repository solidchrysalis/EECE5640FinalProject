import torch
from torch.optim.optimizer import Optimizer
import adagrad_cuda 
import time

class AdagradOptimizer(Optimizer):
    def __init__(self, params, lr=0.01):
        # Initialize the parent Optimizer class with the parameters and learning rate
        defaults = dict(lr=lr)
        self.prev_grads = torch.Tensor([])
        super(AdagradOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            
            for param in group['params']:
                if param.grad is None:
                    continue

                # Get gradients and the parameter (weights)
                grad = param.grad.data
                var = param.data

                if self.prev_grads.numel() == 0:
                    self.prev_grads = torch.zeros(var.numel())

                if var.is_cuda:
                    adagrad_cuda.adagrad(var, grad, lr)
                else:
                    raise RuntimeError("StochasticOptimizer only supports CUDA tensors.")

        return loss
