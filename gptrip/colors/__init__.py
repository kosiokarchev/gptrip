import os
from glob import glob
from itertools import permutations

import numpy as np
from matplotlib import cm, colors


def register(name, clrs):
    cm.register_cmap(name, colors.LinearSegmentedColormap.from_list(name, clrs))
    cm.register_cmap(name+'_r', colors.LinearSegmentedColormap.from_list(name+'_r', clrs[::-1]))


for fname in glob(os.path.join(os.path.dirname(__file__), '*.cmap')):
    register(os.path.basename(fname)[:-5], np.loadtxt(fname, dtype=int)/255.)

for i in permutations(range(3)):
    register(''.join(np.array(('r', 'g', 'b'))[i,]), np.eye(3)[i,])

__all__ = []
