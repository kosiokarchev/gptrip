from functools import partial
from itertools import zip_longest, cycle

import numpy as np
from scipy.signal import stft, get_window

import pandas as pd
from matplotlib import pyplot as plt

from pydub import AudioSegment

from .abc import AbstractSong
from .utils import compose


class Song(AbstractSong):
    def __init__(self, inname, start=0, end=0,
                 specrate=16, overlaprate=float('inf'),
                 window=('kaiser', 14),
                 specsmoothing=4, timesmoothing=0.5):
        self.inname = inname

        self.audio = AudioSegment.from_file(self.inname)
        fs = self.audio.frame_rate
        super().__init__(fs, specrate, specsmoothing, timesmoothing)

        self.start, self.end = start, end if end > start else self.audio.duration_seconds
        self.duration = self.end - self.start

        signal = np.array(self.audio.get_array_of_samples()).reshape(-1, self.audio.channels) / self.audio.max_possible_amplitude
        self.signal = signal[int(start*self.fs) : int(end*self.fs)+1 if end > start else None].T

        self.noverlap = int(self.nperseg // overlaprate)
        self.window = get_window(window, self.nperseg)
        self.fspec, self.t, self.rawspecs = stft(
            self.signal, self.fs, self.window,
            nperseg=self.nperseg, noverlap=self.noverlap)

        self.specs = np.abs(self.rawspecs)**2

        self.specs = np.array([
            compose([
                partial(lambda a, n, ax:
                            a.rolling(n, 0, True, axis=ax).mean() if n else a,
                        n=n, ax=ax)
                for ax, n in ((1, self.specsmoothing),
                              (0, self.ntimesmoothing))
            ])(pd.DataFrame(s)).to_numpy()
            for s in self.specs
        ])

        self.spec = self.specs.mean(0)
        self.rawspec = self.rawspecs.mean(0)
        self.currawspec = None

        self.iter = self.rawiter = None

        self.A = [None]

    def to_frac(self, dt):
        return int(self.duration // dt) + 1

    def normalize(self, initial_smoothing=0.5, maxwindow=1., minwindow=0.5, constpower=1, final_smoothing=0.5, power=2,
                  plot=False):
        initial_smoothing = self.to_spec_count(initial_smoothing)
        maxwindow = self.to_spec_count(maxwindow)
        minwindow = self.to_spec_count(minwindow)
        final_smoothing = self.to_spec_count(final_smoothing)

        signal = np.nansum(self.spec, 0)
        signal = pd.Series(signal / signal.max())

        A = signal.rolling(initial_smoothing, 0, True).mean()
        top = A.rolling(maxwindow, 0, True).max().rolling(maxwindow, 0, True).mean()
        bot = signal.rolling(minwindow, 0, True).min().rolling(minwindow, 0, True).mean()

        w = bot ** constpower
        y = (A - bot) / (top - bot) * (1 - w) + w
        ym = y.rolling(final_smoothing, 0, True).mean()
        # ret = PowerNorm(power, 0)(((A-bot)/(top-bot)*(1-w) + w).rolling(final_smoothing, 0, True).mean().to_numpy()).data
        self.A = np.sin(ym * np.pi / 2) ** power

        if plot:
            plt.plot(np.transpose((signal, A, bot, top, w, y, ym, self.A)))

        return self.A

    def get_iter(self):
        return zip_longest(self.spec.T, self.A)

    def __next__(self):
        self.currawspec = next(self.rawiter)
        return next(self.iter)

    def __iter__(self):
        self.rawiter = iter(self.rawspec.T)
        self.iter = self.get_iter()
        return self


class EndlessSong(Song):
    def get_iter(self):
        return zip(cycle(self.spec.T), cycle(self.A))
