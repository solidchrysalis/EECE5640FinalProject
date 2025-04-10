import torch
from torch.optim.optimizer import Optimizer
import stochastic_cuda 
import time

class StochasticOptimizer(Optimizer):
    def __init__(self, params, lr=0.01):
        # Initialize the parent Optimizer class with the parameters and learning rate
        defaults = dict(lr=lr)
        super(StochasticOptimizer, self).__init__(params, defaults)

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

                if var.is_cuda:
                    stochastic_cuda.stochastic(var, grad, lr)
                else:
                    raise RuntimeError("StochasticOptimizer only supports CUDA tensors.")


        return loss
