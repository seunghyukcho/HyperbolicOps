import torch


@torch.jit.script
def halfplane2disk(x, c):
    c = -c
    a, b = x[..., 0], x[..., 1]
    denominator = c * a.pow(2) + (b + 1).pow(2)
    x = torch.stack([
        c.sqrt() * a.pow(2) + (b.pow(2) - 1) / c.sqrt(),
        -2 * a
    ], dim=-1) / denominator[..., None]

    return x


@torch.jit.script
def disk2halfplane(x, c):
    c = -c
    a, b = x[..., 0], x[..., 1]
    denominator = (c.sqrt() * a - 1).pow(2) + c * b.pow(2)
    x = torch.stack([
        -2 * b,
        1 - (a.pow(2) + b.pow(2)) * c
    ], dim=-1) / denominator[..., None]

    return x


@torch.jit.script
def disk2lorentz(x, c):
    c = -c
    x_norm = x.pow(2).sum(dim=-1, keepdim=True)
    x = torch.concat([
        (1 + x_norm * c) / (c.sqrt() * (1 - x_norm * c)),
        2 * x / (1 - x_norm * c)
    ], dim=-1)
    return x


@torch.jit.script
def lorentz2disk(x, c):
    c = -c
    x = x / (c.sqrt() * x[..., :1] + 1)
    return x[..., 1:]


@torch.jit.script
def lorentz2halfplane(x, c):
    c = -c
    t, a, b = x[..., 0], x[..., 1], x[..., 2]
    x0 = -b / (c.sqrt() * (t - a))
    x1 = 1 / (c.sqrt() * (t - a))
    x = torch.stack([x0, x1], dim=-1)
    return x


@torch.jit.script
def lorentz2halfplane_log(x, c):
    c = -c
    t, a, b = x[..., 0], x[..., 1], x[..., 2]
    x0 = -b / (c.sqrt() * (t - a))
    x1 = -0.5 * c.log() - (t - a).log()
    x = torch.stack([x0, x1], dim=-1)
    return x


@torch.jit.script
def halfplane2lorentz(x, c):
    c = -c
    a, b = x[..., 0], x[..., 1]
    x = torch.stack([
        (1 + c * a.pow(2) + b.pow(2)) / (2 * c.sqrt() * b),
        (-1 + c * a.pow(2) + b.pow(2)) / (2 * c.sqrt() * b),
        -a / b
    ], dim=-1)

    return x


@torch.jit.script
def halfplane2lorentz_log(x, c):
    c = -c
    a, logb = x[..., 0], x[..., 1]
    b, b_inverse = logb.exp(), (-logb).exp()
    x = torch.stack([
        ((1 + c * a.pow(2)) * b_inverse + b) / (2 * c.sqrt()),
        ((-1 + c * a.pow(2)) * b_inverse + b) / (2 * c.sqrt()),
        -a * b_inverse
    ], dim=-1)

    return x

