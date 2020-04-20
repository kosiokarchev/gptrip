from collections import deque

import numpy as np
import sounddevice as sd

from ..song import AbstractSong, EndlessSong


class SDSong(AbstractSong):
    def __init__(self, input_device=None, specrate=16,
                 specsmoothing=4, timesmoothing=0.5):
        if input_device is None:
            input_device = sd.default.device[0]
        self.device_info = sd.query_devices(device=input_device)
        fs = self.device_info['default_samplerate']

        super().__init__(fs, specrate, specsmoothing, timesmoothing)

        self.input_stream = sd.InputStream(
            samplerate=self.fs, blocksize=self.nperseg,
            device=input_device, channels=1, dtype='float32',
            callback=self.callback
        )

        self.frames = deque(maxlen=self.ntimesmoothing)

    def callback(self, indata, nsamples, dt, status):
        self.frames.append(indata.sum(-1).copy())

    def __next__(self):
        return (
            # pd.Series(
                np.mean(abs(np.fft.rfft(list(self.frames))) ** 2, axis=0)
            # ).rolling(self.specsmoothing, 0, True).mean().to_numpy()
            if self.frames else None,
            None
        )

    def __iter__(self):
        self.input_stream.start()
        return self


class EndlessPlayingSong(EndlessSong):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ostream = sd.OutputStream(
            self.fs, self.nperseg, channels=1, dtype='float32',
            callback=self.play
        )

    def play(self, outdata, nframes, dt, status):
        if self.currawspec is not None:
            signal = self.nperseg * np.fft.irfft(self.currawspec)
            signal /= (self.window + 1e-7)
            outdata[:] = signal.reshape(outdata.shape)

    def __iter__(self):
        ret = super().__iter__()
        self.ostream.start()
        return ret