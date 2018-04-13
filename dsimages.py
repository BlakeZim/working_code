import glob
import matplotlib.pyplot as plt
import time
import get_file_list as gfl
plt.ion()

import scipy.misc
import numpy as np

ds = 8

imdirbw = '/home/sci/crottman/korenberg/data/photo/seg_high_res_crop/bw/'
imdircol = '/home/sci/crottman/korenberg/data/photo/seg_high_res_crop/color/'

imdirbwout = '/home/sci/crottman/korenberg/data/photo/seg_low_res_crop' + str(ds) + '/bw/'
imdircolout = '/home/sci/crottman/korenberg/data/photo/seg_low_res_crop' + str(ds) + '/color/'

block = 'block4/'

files = glob.glob(imdirbw + block + '/*.png')
files = sorted(files, key=lambda x: int(x[x.rfind('_')+1:x.rfind('.')]))


for filename in files:
    fname = filename[filename.rfind('/')+1:]
    print fname

    ImArr = plt.imread(filename)
    ImArr = sum([ImArr[i::ds, j::ds]
                 for i in range(ds) for j in range(ds)]) # downsample
    ImArr *= 1/(ds**2*1.0)

    ImArrcol = plt.imread(imdircol + block + fname)
    ImArrcol = sum([ImArrcol[i::ds, j::ds]
                 for i in range(ds) for j in range(ds)]) # downsample
    ImArrcol *= 1/(ds**2*1.0)

    # print ImArrcol.shape, ImArrcol.min(), ImArrcol.max()
    scipy.misc.imsave(imdirbwout + block + fname, ImArr)
    scipy.misc.imsave(imdircolout + block + fname, ImArrcol)
