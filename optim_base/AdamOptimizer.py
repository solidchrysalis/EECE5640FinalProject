import torch
from torch.optim.optimizer import Optimizer
import adam_cuda 
import time

class AdamOptimizer(Optimizer):
    def __init__(self, params, lr=0.01):
        # Initialize the parent Optimizer class with the parameters and learning rate
        defaults = dict(lr=lr)
        self.prev_mom = torch.Tensor([])
        super(AdamOptimizer, self).__init__(params, defaults)

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

                if self.prev_mom.numel() == 0:
                    self.prev_mom = torch.zeros(var.numel())

                if var.is_cuda:
                    adam_cuda.adam(var, grad, self.prev_mom, 0.9, lr)
                else:
                    raise RuntimeError("AdamOptimizer only supports CUDA tensors.")

        return loss
