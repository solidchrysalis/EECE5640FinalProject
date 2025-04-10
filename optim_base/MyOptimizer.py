# Import the necessary libraries 
import torch 
import torch.nn as nn 
  
# MomentumOptimizer 
class MyOptimizer(torch.optim.Optimizer): 
      
    # Init Method: 
    def __init__(self, params, lr=1e-3, momentum=0.9): 
        super(MomentumOptimizer, self).__init__(params, defaults={'lr': lr}) 
        self.momentum = momentum 
        self.state = dict() 
        for group in self.param_groups: 
            for p in group['params']: 
                self.state[p] = dict(mom=torch.zeros_like(p.data)) 
      
    # Step Method 
    def step(self): 
        pass