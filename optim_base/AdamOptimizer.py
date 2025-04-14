import torch
from torch.optim.optimizer import Optimizer
import adam_cuda 
import time

class AdamOptimizer(Optimizer):
    def __init__(self, params, device=None, lr=0.01):
        # Initialize the parent Optimizer class with the parameters and learning rate
        defaults = dict(lr=lr)
        self.prev_mom = dict()
        self.device = device
        if self.device is None:
            raise RuntimeError("No device found")
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
                curr_layer_mom = None

                # Update dictionary based the parameter (the layer) so we have custom momentums for each layer
                if param not in self.prev_mom.keys():
                    self.prev_mom[param] = torch.zeros_like(var).to(self.device)

                curr_layer_mom = self.prev_mom[param]

                if var.is_cuda:
                    adam_cuda.adam(var, grad, curr_layer_mom, 0.9, lr)
                else:
                    raise RuntimeError("AdamOptimizer only supports CUDA tensors.")

        return loss
