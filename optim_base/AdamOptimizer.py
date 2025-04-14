import torch
from torch.optim.optimizer import Optimizer
import adam_cuda 
import time

class AdamOptimizer(Optimizer):
    def __init__(self, params, device=None, lr=0.005):
        # Initialize the parent Optimizer class with the parameters and learning rate
        defaults = dict(lr=lr)
        self.prev_mean = dict()
        self.prev_variance = dict()
        self.device = device
        self.epoch = 1.0
        self.epsilon = 1e-6
        self.beta_1 = 0.9
        self.beta_2 = 0.9
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
                curr_layer_mean = None
                curr_layer_variance = None

                # Update dictionary based the parameter (the layer) so we have custom momentums for each layer
                if param not in self.prev_mean.keys():
                    self.prev_mean[param] = torch.zeros_like(var).to(self.device)

                if param not in self.prev_variance.keys():
                    self.prev_variance[param] = torch.zeros_like(var).to(self.device)

                # Magic numbs are beta 1 and 2 - todo fix them
                if var.is_cuda:
                    adam_cuda.adam(var, grad, curr_layer_mean, curr_layer_variance, self.beta_1, self.beta_2, self.epoch, self.epsilon, lr)
                else:
                    raise RuntimeError("AdamOptimizer only supports CUDA tensors.")

        self.epoch += 1
        return loss
