import torch
import math
from torch.distributions import constraints
from torch.distributions import Distribution
from torch.distributions.transforms import PowerTransform
from torch.distributions.gamma import Gamma
from torch.distributions.transformed_distribution import TransformedDistribution


class InverseGamma(TransformedDistribution):
    """
    Creates an inverse-gamma distribution parameterized by
    `concentration` and `rate`.

        X ~ Gamma(concentration, rate)
        Y = 1/X ~ InverseGamma(concentration, rate)

    :param torch.Tensor concentration: the concentration parameter (i.e. alpha).
    :param torch.Tensor rate: the rate parameter (i.e. beta).
    """
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.positive
    has_rsample = True

    def __init__(self, concentration, rate, validate_args=None):
        base_dist = Gamma(concentration, rate)
        super().__init__(
            base_dist,
            PowerTransform(-base_dist.rate.new_ones(())),
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(InverseGamma, _instance)
        return super().expand(batch_shape, _instance=new)


    @property
    def concentration(self):
        return self.base_dist.concentration

    @property
    def rate(self):
        return self.base_dist.rate
    
def _matmul(x, S, y):
    return (x[...,None,:]@S@y[...,None])[...,0,0]

def _safe_log(x):
    return x.clamp(min=torch.finfo(x.dtype).eps).log()
    
class CircularProjectedNormal(Distribution):
    arg_constraints = {
        "loc": constraints.real_vector,
        "scale": constraints.real_vector,
    }
    support = constraints.real_vector
    has_rsample = False

    def __init__(self, loc, scale, validate_args=None):
        assert loc.dim() >= 1
        assert scale.dim() >= 2\

        self.loc = loc
        self.scale = scale
        batch_shape = loc.shape[:-1]
        event_shape = loc.shape[-1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        S_det = torch.linalg.det(scale)
        self.S_inv = torch.linalg.inv(scale)
        self.C = 0.5*_matmul(loc, self.S_inv, loc) + 0.5*S_det.log()

    def log_prob(self, u):
        if self._validate_args:
            event_shape = u.shape[-1:]
            if event_shape != self.event_shape:
                raise ValueError(
                    f"Expected event shape {self.event_shape}, "
                    f"but got {event_shape}"
                )
            self._validate_sample(u)

        A = _matmul(u, self.S_inv, u)
        B = _matmul(u, self.S_inv, self.loc)
        D = B/A.sqrt()

        Phi = 0.5*(1 + torch.erf(D*0.5**0.5)) 
        phi = 1/math.sqrt(2*torch.pi)*torch.exp(-0.5*D**2)
        ll = -math.log(2*torch.pi) - torch.log(A) - self.C + torch.log(1 + D*Phi/phi)
        return ll