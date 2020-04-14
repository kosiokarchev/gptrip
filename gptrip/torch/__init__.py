import sys

try:
    import torch
    from .gptrip import GPTripTorch
    from . import utils
except ModuleNotFoundError:
    sys.stderr.write('Torch not found.')
