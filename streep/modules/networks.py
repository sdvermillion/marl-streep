from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

class MLP(nn.Module):
    def __init__(self, dims: List[int], zero_init: bool = False, output_activation: Optional[Callable] = None) -> None:
        super(MLP, self).__init__()
        assert len(dims) >= 2
        self.network = nn.ModuleList([])
        for i in range(len(dims)-1):
            layer = nn.Linear(dims[i], dims[i+1])
            if zero_init:
                layer.weight.data.zero_()
                layer.bias.data.zero_()
            self.network.append(layer)
        self.output_activation = output_activation

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.network):
            x = layer(x)
            if i < len(self.network)-1:
                x = F.relu(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x
    
class DistParams(nn.Module):
    def __init__(self, hdim: int, zdim: int) -> None:
        super(DistParams, self).__init__()
        self.alpha = nn.Linear(hdim, zdim)
        self.beta = nn.Linear(hdim, zdim)

        self.alpha.weight.data.zero_()
        self.alpha.bias.data.fill_(1.)

        self.beta.weight.data.zero_()
        self.beta.bias.data.fill_(1.)

    def forward(self, x: Tensor) -> Tuple[Tensor,Tensor]:
        alpha = F.softplus(self.alpha(x))
        beta = F.softplus(self.beta(x))
        return alpha, beta
