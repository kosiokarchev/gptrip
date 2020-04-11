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

    @staticmethod
    def fftshift(img):
        return img.roll([s//2 for s in img.shape],
                        dims=tuple(range(img.ndim)))

    @copy_signature(GPTrip.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.spec = to_tensor(self.spec)

        self.fimg = to_tensor(self.fimg)
        self.ximg_filled = to_tensor(self.ximg_filled)
        self.xspec_masked = to_tensor(self.xspec_masked)

        self.ipol = FixedXInterp1d(self.xspec_masked, self.ximg_filled)

    @copy_signature(GPTrip.normalize)
    def normalize(self, *args, **kwargs):
        return to_tensor(super().normalize(*args, **kwargs))

    def generate_image(self, spec, seeds):
        amps = torch.sqrt(self.ipol(spec[~self.xspec.mask]) / self.fimg)
        amps[torch.isnan(amps)] = 0.

        img = torch.irfft(seeds * amps.unsqueeze(-1), 2,
                          signal_sizes=(self.height, self.width))
        return img, amps
