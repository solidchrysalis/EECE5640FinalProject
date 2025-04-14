import torch
from torch.optim.optimizer import Optimizer
import adagrad_cuda 
import time

class AdagradOptimizer(Optimizer):
    def __init__(self, params, device=None, lr=0.005):
        # Initialize the parent Optimizer class with the parameters and learning rate
        defaults = dict(lr=lr)
        self.prev_grads = dict()
        self.device = device
        self.epsilon = 1e-8
        if self.device is None:
            raise RuntimeError("No device found")
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
                curr_layer_grads = None

                if param not in self.prev_grads.keys():
                    self.prev_grads[param] = torch.full_like(var, 1e-6).to(self.device)

                curr_layer_grads = self.prev_grads[param]

                if var.is_cuda:
                    adagrad_cuda.adagrad(var, grad, curr_layer_grads, self.epsilon, lr)
                else:
                    raise RuntimeError("StochasticOptimizer only supports CUDA tensors.")

        return loss
