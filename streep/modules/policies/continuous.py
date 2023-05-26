from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta

from . import BasePolicy
from ..networks import MLP, DistParams

Tensor = torch.Tensor

class BaseContinuousPolicy(BasePolicy):
    def __init__(self):
        super(BaseContinuousPolicy, self).__init__()

    def get_action(self, state:Tensor, reparameterize:bool = False)->Tensor:
        dist = self.get_policy(state)
        if reparameterize:
            action = dist.rsample()
        else:
            action = dist.sample()
        return action
    

class BaseBetaPolicy(BaseContinuousPolicy):
    def __init__(self):
        super(BaseBetaPolicy, self).__init__()

    def get_policy(self, state:Tensor)->Beta:
        alpha, beta = self.forward(state)
        return Beta(concentration1=alpha, concentration0=beta)
    
    def forward(self, state:Tensor)->Tuple[Tensor,Tensor]:
        x = self.encoder(state)
        alpha, beta = self.decoder(x)
        return alpha, beta
    

class MLPBetaPolcy(BaseBetaPolicy):
    def __init__(self, state_dim:int, hdim:List[int], action_dim:int):
        super(MLPBetaPolcy, self).__init__()
        self.encoder = MLP(dims = [state_dim, *hdim])
        self.decoder = DistParams(hdim[-1], action_dim)