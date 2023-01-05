import torch
import geoopt
from numbers import Number
from torch.nn import functional as F
from torch.distributions import Normal, MultivariateNormal


class HyperbolicWrappedNormal(torch.distributions.Distribution):

    arg_constraints = {
        'loc': torch.distributions.constraints.real,
        'scale': torch.distributions.constraints.positive
    }
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        raise NotImplementedError

    def __init__(self, loc, scale, c=1, validate_args=None):
        self.loc = loc
        self.scale = scale
        self.c = c
        self.manifold = geoopt.manifolds.Lorentz(1 / self.c)
        self.dim = self.loc.size(-1) - 1
       
        self.base = Normal(
            torch.zeros_like(self.scale),
            self.scale
        )

        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape, event_shape = torch.Size(), torch.Size()
        else:
            batch_shape = self.loc.shape[:-1]
            event_shape = torch.Size([self.dim])
        super(HyperbolicWrappedNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, z):
        u = self.manifold.logmap(self.mean, z)
        v = self.manifold.transp(self.mean, self.origin, u)
        log_prob_v = self.base.log_prob(v[:, :, 1:]).sum(-1)

        r = self.manifold.norm(u)
        log_det = (self.latent_dim - 1) * (torch.sinh(r).log() - r.log())

        log_prob_z = log_prob_v - log_det
        return log_prob_z

    def rsample(self, N):
        v = self.base.rsample([N])
        v = F.pad(v, (1, 0))

        u = self.manifold.transp0(self.mean, v)
        z = self.manifold.expmap(self.mean, u)

        return z

    def sample(self, N):
        with torch.no_grad():
            return self.rsample(N)

