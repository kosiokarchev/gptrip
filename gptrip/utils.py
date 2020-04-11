from collections.abc import Iterable, Reversible
import numpy as np


def flatten(*lst):
    for item in lst:
        yield from (flatten(*item) if isinstance(item, Iterable) and not isinstance(item, str) else (item,))


def compose(*funcs):
    def ret(arg):
        fncs = flatten(funcs)
        for f in reversed(fncs if isinstance(fncs, Reversible) else list(fncs)):
            arg = f(arg)
        return arg

    return ret


def rfftfreqn(shape, d=1):
    return np.sqrt(np.power(
        np.array(np.meshgrid(*[np.fft.fftfreq(s) for s in shape[:-1]], np.fft.rfftfreq(shape[-1]), indexing='ij')) / d,
        2).sum(0)
    )


def image_to_rawvideo(img):
    return (np.array(img[..., :3] * 255)).astype(np.uint8).tobytes()


def mask_out_of_range(ma: np.ma.masked_array):
    ma.mask = np.broadcast_to(ma.mask, ma.shape)
    ma.mask[np.logical_or(ma < 0, ma > 1)] = True
    ma.fill_value = np.nan
    return ma


class RegularXInterpolator:
    def __init__(self, low, high, y):
        self.low, self.step = low, (high - low) / (len(y) - 1)
        self.y = y

    def __call__(self, x):
        x = x - self.low
        i = int(x // self.step)
        alpha = x/self.step - i
        return alpha * self.y[i+1] + (1-alpha) * self.y[i]
