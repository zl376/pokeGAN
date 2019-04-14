# Utilities
# Author: Zhe Liu (zl376@cornell.edu)
# Date: 2018-08-26

import matplotlib.pyplot as plt
import numpy as np


        
# Plot
def plots(ims, figsize=(12,6), 
               rows=1, 
               scale=None, 
               interp=False, 
               titles=None):
    
    if scale != None:
        lo, hi = scale
        ims = ims.copy()
        ims[ims > hi] = hi
        ims[ims < lo] = lo
        ims = (ims - lo)/(hi - lo) * 1.0
        
    if ims.ndim == 2:
        ims = ims[np.newaxis, ..., np.newaxis];
    elif ims.ndim == 3:
        ims = ims[..., np.newaxis];
    if ims.shape[-1] == 1:
        ims = np.tile(ims, (1,1,1,3))
    #ims = ims.astype(np.uint8)
    #print(ims.shape)
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
