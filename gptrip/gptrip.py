from math import pi
from time import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import ffmpeg

from .abc import AbstractGPTrip
from .song import Song
from .utils import image_to_rawvideo, RegularXInterpolator, compose


class GPTrip(AbstractGPTrip):
    @staticmethod
    def fftshift(img):
        return np.fftshift(img)

    def __init__(self, song: Song, width, height,
                 minfreq=60., maxfreq=1e4,
                 minwav=0., maxwav=1.,
                 random_directions=False, random_seeds=False):
        self.song = song
        self.nframes = len(self.song.spec.T)

        super().__init__(width, height, self.song.fspec,
                         minfreq, maxfreq, minwav, maxwav,
                         random_directions, random_seeds)

    def interpolate_seed(self, dt):
        self.seed = RegularXInterpolator(0, self.nframes, self.random((self.song.to_frac(dt),) + self.fimg.shape))

    def interpolate_directions(self, dt, spread):
        self.directions = compose(
            self.phases_to_directions,
            RegularXInterpolator(0, self.nframes,
                                 spread * 2*pi * (self.random((self.song.to_frac(dt),) + self.fimg.shape)) - 0.5))

    def render(self, do_plot=False, fftshift=False,
               cmap=plt.get_cmap(), norm=plt.Normalize(-1, 1),
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

        for i, (spec, a) in enumerate(self.song):
            try:
                t0 = time()
                spec = self.to_array(spec)
                img, power = self.generate_image(spec, self.get_seeds(i))

                if fftshift:
                    img = self.fftshift(img)

                sigma = self.math.sqrt((img ** 2).mean())
                img_final = cmap(norm(img / sigma))[..., :3]
                img_final = self.clip(img_final if a is None else a * img_final, 0., 1.)

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
