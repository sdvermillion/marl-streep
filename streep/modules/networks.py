import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, dims: list[int], zero_init: bool = False, output_activation=None):
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

    def forward(self, x):
        for i, layer in enumerate(self.network):
            x = layer(x)
            if i < len(self.network)-1:
                x = F.relu(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x