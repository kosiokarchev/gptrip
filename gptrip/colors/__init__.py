import os
from glob import glob

import numpy as np
from matplotlib import cm, colors


for fname in glob(os.path.join(os.path.dirname(__file__), '*.cmap')):
    cm.register_cmap(cmap=colors.ListedColormap(
        colors=np.loadtxt(fname, dtype=int)/255.,
        name=os.path.basename(fname)[:-5]
    ))

__all__ = []
