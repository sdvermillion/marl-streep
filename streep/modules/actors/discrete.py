from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from . import BaseActor
from ..networks import MLP

Tensor = torch.Tensor

class BaseDiscreteActor(BaseActor):
    def __init__(self):
        super(BaseDiscreteActor, self).__init__()

    def get_policy(self, state: Tensor)->Categorical:
        action_probs = self.forward(state)
        return Categorical(action_probs)

    def get_action(self, state: Tensor, deterministic: bool = True) -> Tensor:
        dist = self.get_policy(state)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
        return action
    

class MLPDiscreteActor(BaseDiscreteActor):
    def __init__(self, state_dim: int, hdim: List[int], action_dim: int):
        super(MLPDiscreteActor, self).__init__()
        self.network = MLP(dims=[state_dim, *hdim, action_dim], zero_init=True)
    
    def forward(self, state: Tensor) -> Tensor:
        action_scores = self.network(state)
        action_probs = F.softmax(action_scores, dim=-1)
        return action_probs