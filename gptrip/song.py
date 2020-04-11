import os
from operator import itemgetter
from functools import partial

import numpy as np
from scipy.signal import spectrogram

import pandas as pd

from pydub import AudioSegment

from .utils import compose


# def from_file(fname):
#     if os.path.splitext(fname)[1] == '.npz':
#         fs, data = itemgetter('fs', 'data')(np.load(fname))
#         return AudioSegment(data=data.tobytes(), channels=)
#     else:
#         return AudioSegment.from_file(fname)


class Song:
    def __init__(self, inname, start=0, end=0,
                 specrate=16, overlaprate=float('inf'),
                 specsmoothing=4, timesmoothing=0.5,
                 ):
        self.inname = inname

        self.audio = AudioSegment.from_file(self.inname)
        self.start, self.end = start, end if end > start else self.audio.duration_seconds
        self.duration = self.end - self.start

        self.fs = self.audio.frame_rate
        signal = np.array(self.audio.get_array_of_samples()).reshape(-1, self.audio.channels) / self.audio.max_possible_amplitude
        self.signal = signal[int(start*self.fs) : int(end*self.fs)+1 if end > start else None].T

        nperseg = int(self.fs // specrate)
        noverlap = int(nperseg // overlaprate)
        self.fspec, self.t, self.spec = spectrogram(self.signal, self.fs, nperseg=nperseg, noverlap=noverlap)
        self.dt = self.t[1] - self.t[0]
        self.specrate = 1 / self.dt

        self.spec = np.array([
            compose([
                partial(lambda a, n, ax:
                            a.rolling(n, 0, True, axis=ax).mean() if n else a,
                        n=n, ax=ax)
                for ax, n in ((1, specsmoothing),
                              (0, self.to_spec_count(timesmoothing)))
            ])(pd.DataFrame(s)).to_numpy()
            for s in self.spec
        ])

    def to_spec_count(self, dt):
        return int(dt * self.specrate)

    def to_frac(self, dt):
        return int(self.duration // dt) + 1