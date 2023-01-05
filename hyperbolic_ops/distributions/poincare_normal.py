import torch
import geoopt
from numbers import Number

from .hyperbolic_radius import HyperbolicRadius
from .hyperbolic_uniform import HypersphericalUniform

MIN_NORM = 1e-15


def expmap_polar(c, x, u, r, dim: int = -1):
    m = geoopt.manifolds.PoincareBall(1.0)
    sqrt_c = c.sqrt()
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = (
        (sqrt_c / 2 * r).tanh()
        * u
        / (sqrt_c * u_norm)
    )

    gamma_1 = m.mobius_add(x, second_term)
    return gamma_1


class PoincareNormal(torch.distributions.Distribution):
    arg_constraints = {
        'loc': torch.distributions.constraints.interval(-1, 1), 
        'scale': torch.distributions.constraints.positive
    }
    support = torch.distributions.constraints.interval(-1, 1)
    has_rsample = True

    @property
    def mean(self):
        return self.loc
    
    def __init__(self, loc, scale, c=1, validate_args=None):
        assert not (torch.isnan(loc).any() or torch.isnan(scale).any())
        self.loc = loc
        self.scale = scale  # .clamp(min=0.1, max=7.)
        self.c = c
        self.manifold = geoopt.manifolds.PoincareBall(self.c)
        self.dim = self.loc.size(-1)

        self.radius = HyperbolicRadius(self.dim, self.manifold.c, self.scale)
        self.direction = HypersphericalUniform(self.dim - 1, device=loc.device)

        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape, event_shape = torch.Size(), torch.Size()
        else:
            batch_shape = self.loc.shape[:-1]
            event_shape = torch.Size([self.dim])

        super(PoincareNormal, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        alpha = self.direction.sample(torch.Size([*shape[:-1]]))
        radius = self.radius.rsample(sample_shape)
        res = expmap_polar(self.manifold.c, self.loc, alpha, radius)
        return res

    def log_prob(self, value):
        loc = self.loc.expand(value.shape)
        radius_sq = self.manifold.dist(loc, value, keepdim=True).pow(2)
        res = - radius_sq / 2 / self.scale.pow(2) - self.direction._log_normalizer() - self.radius.log_normalizer
        return res

