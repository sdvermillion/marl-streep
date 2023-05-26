import torch
import torch.nn as nn

Tensor = torch.Tensor

class BasePolicy(nn.Module):
    def __init__(self):
        super(BasePolicy, self).__init__()

    def get_log_prob(self, state:Tensor, action:Tensor)->Tensor:
        return self.get_policy(state).log_prob(action)
    
    def get_entropy(self, state:Tensor)->Tensor:
        return self.get_policy(state).entropy()