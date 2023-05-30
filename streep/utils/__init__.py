import torch
from torchaudio.functional import lfilter

Tensor = torch.Tensor

def discounted_cumsum(x: Tensor, discount: float) -> Tensor:
    a_coeff = Tensor([1, -discount])
    b_coeff = Tensor([1, 0])
    return lfilter(x.flip(dims = (0,)), a_coeff, b_coeff, clamp = False).flip(dims = (0,))