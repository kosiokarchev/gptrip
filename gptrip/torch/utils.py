import numpy as np
import torch
from matplotlib.pyplot import get_cmap


# Hack into torch to automatically copy to cpu when converting to numpy
__Tensor_numpy = torch.Tensor.numpy
torch.Tensor.numpy = lambda self: __Tensor_numpy(self.cpu())


def to_tensor(a):
    return torch.tensor(a, dtype=torch.get_default_dtype(), device=None)


class Normalize:
    def __init__(self, vmin=None, vmax=None):
        self.vmin = vmin
        self.vmax = vmax

    def autoscale(self, value):
        self.vmin = value.min()
        self.vmax = value.max()

    def process_value(self, value):
        return (value - self.vmin) / (self.vmax - self.vmin)

    def __call__(self, value):
        if self.vmin is None:
            self.autoscale(value)

        return self.process_value(value)


class Colormap:
    def __call__(self, value: torch.tensor) -> torch.tensor:
        raise NotImplementedError


class PltColormap(Colormap):
    def __init__(self, *args, **kwargs):
        self.cmap = get_cmap(*args, **kwargs)

    def __call__(self, value: torch.tensor):
        return to_tensor(self.cmap(value))


class ListedColormap(Colormap):
    def __init__(self, vals, low=None, high=None, invalid=0.):
        self.n = len(vals)
        self.vals = to_tensor(np.concatenate((
            np.broadcast_to(invalid, vals[:1].shape),
            np.broadcast_to(vals[0] if low is None else low, vals[:1].shape),
            vals,
            np.broadcast_to(vals[-1] if high is None else high, vals[:1].shape)
        ), 0))

        self.mod = 1/self.n
        self.zero = torch.tensor(0, dtype=torch.long)

    def __call__(self, value: torch.tensor):
        idx = torch.clamp(value, -self.mod, 1.) // self.mod + 2
        return self.vals[torch.where(torch.isnan(idx), self.zero, idx.long())]


class SampledColormap(ListedColormap):
    def __init__(self, *cmap_args, n=256, low=None, high=None, invalid=0., **cmap_kwargs):
        super().__init__(
            get_cmap(*cmap_args, **cmap_kwargs)(np.linspace(1/n, 1-1/n, n)),
            low=low, high=high, invalid=invalid
        )


class FixedXInterp1d:
    def __init__(self, xold, xnew):
        device = xold.device
        self.xold = xold
        self.xnew = xnew

        d = (xold.unsqueeze(-2).to('cpu') - xnew.unsqueeze(-1).to('cpu'))
        self.it = torch.where(d < 0, torch.scalar_tensor(float('inf'), dtype=xold.dtype, device='cpu'), d).argmin(-1).to(device)
        self.ib = torch.where(d > 0, torch.scalar_tensor(-float('inf'), dtype=xold.dtype, device='cpu'), d).argmax(-1).to(device)

        xt, xb = xold[self.it], xold[self.ib]
        self.a = (xnew - xb) / (xt - xb)

    def __call__(self, yold):
        yt, yb = yold[self.it], yold[self.ib]
        return self.a * (yt - yb) + yb