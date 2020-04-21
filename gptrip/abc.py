from math import pi
import numpy as np

from scipy.interpolate import interp1d

from matplotlib.colors import LogNorm
from.utils import rfftfreqn, mask_out_of_range


class AbstractSong:
    def __init__(self, fs, specrate, specsmoothing=4, timesmoothing=0.5):
        self.fs = fs
        self.nperseg = int(fs // specrate)
        self.specrate = fs / self.nperseg

        self.fspec = np.fft.rfftfreq(self.nperseg, 1/self.fs)

        self.specsmoothing = specsmoothing
        self.ntimesmoothing = self.to_spec_count(timesmoothing)

    def to_spec_count(self, dt):
        return int(dt * self.specrate)

    def __next__(self):
        raise NotImplementedError

    def __iter__(self):
        return self


class AbstractGPTrip:
    math = np
    random = np.random.random
    clip = np.clip
    to_array = np.array

    def __init__(self, width, height, fspec,
                 minfreq=60, maxfreq=1e4, minwav=0., maxwav=1.,
                 random_directions=False, random_seeds=False):
        self.width, self.height = width, height
        self.unit = max(width, height)
        self.d2f = 1 / (width * height)

        self.fimg = rfftfreqn((height, width))
        self.ximg = mask_out_of_range(
            LogNorm(1 / (maxwav * self.unit), 1 / (np.clip(minwav, np.sqrt(2) / self.unit, maxwav) * self.unit))(
                self.fimg)
        )
        self.ximg_filled = self.to_array(self.ximg.filled())

        self.xspec = mask_out_of_range(LogNorm(minfreq, maxfreq)(fspec))
        self.xspec_masked = self.to_array(self.xspec[~self.xspec.mask])

        self.fimg = self.to_array(self.fimg)

        self.random_directions = random_directions
        self._directions = None
        self.random_seeds = random_seeds
        self._seed = None

    def coherent(self):
        self._directions = self.to_array([1., 0.])

    def exact(self, scatter=0.):
        self._seed = self.to_array(np.clip(0.42 + np.random.normal(0., scatter, self.fimg.shape), 1e-3, 1.-1e-3))

    def seed(self, i):
        if self.random_seeds or self._seed is None:
            self._seed = self.random(self.fimg.shape)
        return self._seed

    def phases_to_directions(self, phases):
        return self.math.stack((self.math.cos(phases),
                                self.math.sin(phases)), axis=-1)

    def directions(self, i):
        if self.random_directions or self._directions is None:
            self._directions = self.phases_to_directions(2*pi * self.random(self.fimg.shape))
        return self._directions

    def get_seeds(self, i):
        return self.directions(i) * self.math.sqrt(-2 * self.math.log(1-self.seed(i)) * self.d2f)[..., None]

    def generate_image(self, spec, seeds):
        ipol = interp1d(self.xspec_masked, spec[~self.xspec.mask],
                        axis=0, bounds_error=False, fill_value=0.)
        amps = np.nan_to_num(np.sqrt(ipol(self.ximg_filled) / self.fimg))
        img = np.fft.irfft2((seeds[..., 0] + seeds[..., 1]*1j) * amps)
        return img, amps
