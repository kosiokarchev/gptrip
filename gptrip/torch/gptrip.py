from inspect import signature

def copy_signature(source_fct):
    def copy(target_fct):
        target_fct.__signature__ = signature(source_fct)
        return target_fct
    return copy

import torch

from ..gptrip import GPTrip
from .utils import to_tensor, FixedXInterp1d

class GPTripTorch(GPTrip):
    math = torch
    random = torch.rand
    clip = torch.clamp
    to_array = staticmethod(to_tensor)

    @staticmethod
    def fftshift(img):
        return img.roll([s//2 for s in img.shape],
                        dims=tuple(range(img.ndim)))

    @copy_signature(GPTrip.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ipol = FixedXInterp1d(self.xspec_masked, self.ximg_filled)

    def generate_image(self, spec, seeds):
        amps = torch.sqrt(self.ipol(spec[~self.xspec.mask]) / self.fimg)
        amps[torch.isnan(amps)] = 0.

        img = torch.irfft(seeds * amps.unsqueeze(-1), 2,
                          signal_sizes=(self.height, self.width))
        return img, amps
