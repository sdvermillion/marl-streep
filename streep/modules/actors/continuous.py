from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta

from . import BaseActor
from ..networks import MLP, DistParams

Tensor = torch.Tensor

class BaseContinuousActor(BaseActor):
    def __init__(self):
        super(BaseContinuousActor, self).__init__()

    def get_action(self, state: Tensor, reparameterize: bool = False) -> Tensor:
        dist = self.get_policy(state)
        if reparameterize:
            action = dist.rsample()
        else:
            action = dist.sample()
        return action
    

class BaseBetaActor(BaseContinuousActor):
    def __init__(self):
        super(BaseBetaActor, self).__init__()

    def get_policy(self, state: Tensor) -> Beta:
        alpha, beta = self.forward(state)
        return Beta(concentration1=alpha, concentration0=beta)
    
    def forward(self, state: Tensor) -> Tuple[Tensor,Tensor]:
        x = self.encoder(state)
        alpha, beta = self.decoder(x)
        return alpha, beta
    

class MLPBetaActor(BaseBetaActor):
    def __init__(self, state_dim: int, hdim: List[int], action_dim: int):
        super(MLPBetaActor, self).__init__()
        self.encoder = MLP(dims = [state_dim, *hdim])
        self.decoder = DistParams(hdim[-1], action_dim)