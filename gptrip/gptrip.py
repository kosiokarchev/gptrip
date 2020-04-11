from math import pi
from time import time

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import rayleigh


from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import ffmpeg

from .song import Song
from .utils import rfftfreqn, mask_out_of_range, image_to_rawvideo, RegularXInterpolator, compose

class GPTrip:
    math = np
    random = np.random.random
    clip = np.clip

    @staticmethod
    def fftshift(img):
        return np.fftshift(img)

    def __init__(self, song: Song, width, height,
                 minfreq=60., maxfreq=1e4,
                 minwav=0., maxwav=1.,
                 random_directions=False, random_seeds=False):
        self.song = song
        self.spec = song.spec.mean(0)
        self.nframes = len(self.spec.T)

        self.A = np.nansum(self.spec, 0)
        self.A = pd.Series(self.A / self.A.max())

        self.width, self.height = width, height
        self.unit = max(width, height)
        self.d2f = 1 / (width * height)

        self.fimg = rfftfreqn((height, width))
        self.ximg = mask_out_of_range(
            LogNorm(1/(maxwav * self.unit), 1/(np.clip(minwav, np.sqrt(2)/self.unit, maxwav) * self.unit))(self.fimg)
        )
        self.ximg_filled = self.ximg.filled()

        self.xspec = mask_out_of_range(LogNorm(minfreq, maxfreq)(song.fspec))
        self.xspec_masked = self.xspec[~self.xspec.mask]

        self.random_directions = random_directions
        self._directions = None
        self.random_seeds = random_seeds
        self._seed = None

    def normalize(self, initial_smoothing=0.5, maxwindow=1., minwindow=0.5, constpower=1, final_smoothing=0.5, power=2, plot=False):
        initial_smoothing = self.song.to_spec_count(initial_smoothing)
        maxwindow = self.song.to_spec_count(maxwindow)
        minwindow = self.song.to_spec_count(minwindow)
        final_smoothing = self.song.to_spec_count(final_smoothing)

        A = self.A.rolling(initial_smoothing, 0, True).mean()
        top = A.rolling(maxwindow, 0, True).max().rolling(maxwindow, 0, True).mean()
        bot = self.A.rolling(minwindow, 0, True).min().rolling(minwindow, 0, True).mean()

        w = bot**constpower
        y = (A-bot)/(top-bot)*(1-w) + w
        ym = y.rolling(final_smoothing, 0, True).mean()
        # ret = PowerNorm(power, 0)(((A-bot)/(top-bot)*(1-w) + w).rolling(final_smoothing, 0, True).mean().to_numpy()).data
        ret = np.sin(ym * np.pi/2) ** power

        if plot:
            plt.plot(np.transpose((self.A, A, bot, top, w, y, ym, ret)))

        return ret

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

    def interpolate_seed(self, dt):
        self.seed = RegularXInterpolator(0, self.nframes, self.random((self.song.to_frac(dt),) + self.fimg.shape))

    def interpolate_directions(self, dt, spread):
        self.directions = compose(
            self.phases_to_directions,
            RegularXInterpolator(0, self.nframes,
                                 spread * 2*pi * (self.random((self.song.to_frac(dt),) + self.fimg.shape)) - 0.5))

    def generate_image(self, spec, seeds):
        ipol = interp1d(self.xspec_masked, spec[~self.xspec.mask],
                        axis=0, bounds_error=False, fill_value=0.)
        amps = np.nan_to_num(np.sqrt(ipol(self.ximg_filled) / self.fimg))
        img = np.fft.irfft2((seeds[:, 0] + seeds[:, 1]*1j) * amps)
        return img, amps

    def render(self, do_plot=False, fftshift=False,
               A=None, cmap=plt.get_cmap(), norm=plt.Normalize(),
               vidname=None, crf=21):
        if do_plot:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4),
                                    gridspec_kw=dict(width_ratios=(1, 1, 2)))
            l, l1 = axs[0].plot([], [], [], [])
            plt.tight_layout()

        if vidname:
            cmd = ffmpeg.output(
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24',
                             s=f'{self.width}x{self.height}', r=self.song.specrate),
                ffmpeg.input(self.song.inname, ss=self.song.start, to=self.song.end).audio,
                vidname, **{'c:v': 'libx264'}, pix_fmt='yuv420p', crf=crf
            )
            print(cmd.compile())
            writer = cmd.run_async(pipe_stdin=True, overwrite_output=False)

        for i, spec in enumerate(self.spec.T):
            try:
                t0 = time()
                img, power = self.generate_image(spec, self.get_seeds(i))

                if fftshift:
                    img = self.fftshift(img)

                sigma = self.math.sqrt((img ** 2).mean())
                img_final = cmap(norm(img / sigma))[..., :3]
                img_final = self.clip(img_final if A is None else A[i] * img_final, 0., 1.)

                if do_plot:
                    l.set_data(self.xspec, spec)
                    axs[0].relim(); axs[0].autoscale()
                    axs[1].clear()
                    axs[1].imshow(power)
                    axs[2].clear()
                    axs[2].imshow(img_final)
                    plt.pause(1e-4)

                if vidname:
                    writer.stdin.write(image_to_rawvideo(img_final))

                if not bool(vidname):
                    print(i + 1, len(self.song.t), 1 / (time() - t0))
            except KeyboardInterrupt:
                break

        if vidname:
            writer.stdin.close()
            writer.wait()
