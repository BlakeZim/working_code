import matplotlib
matplotlib.use('GTK')
# import Tinker
# matplotlib.use('tkagg')
# import PyCACalebExtras.SetBackend
import matplotlib.pyplot as plt
# plt = PyCACalebExtras.SetBackend.SetBackend('agg')
from mpl_toolkits.mplot3d import Axes3D
import PyCA.Core as ca
import gc
import numpy as np
import sys
import PyCAApps as apps
import csv
import PyCACalebExtras.Common as cc
import PyCACalebExtras.Display as cd
plt.ion()
plt.close('all')
import PyCA.Common as common
from PyCAApps import ElastReg
import matplotlib.mlab as mlab
import scipy
import matplotlib.patches as patches

memT = ca.MEM_DEVICE

conDir = '/local/blakez/Backup/M13_01/data/microscopy/confocal/M13_01_B2_43_S5/zslice_tiles_00/C02/'

#micDir = '/local/blakez/korenbergNAS/3D_database/Raw/Microscopic/Confocal/M13/M13-01-B2-10-S3/2016_10_13_17_45_10--M13-01-B2-10-S3 5x tiling/'
#micOut = '/home/sci/blakez/M13_01/results/microscopic/intensityCorrected/test/'

image = cc.LoadTIFF(conDir + 'M13-01-B2-43-S5-C2_Z0_tile_000.tif',memT)

stack = np.zeros((512,512,420),dtype='uint8')

for ii in range(0,420):
    image = plt.imread(conDir + 'M13-01-B2-43-S5-C2_Z0_tile_{0}.tif'.format(str(int(ii)).zfill(3)),memT)
    stack[:,:,ii] = image

fMask = (stack > 1) # Can Adjust what is considered background by changing the comparison number
f = stack*fMask
m = np.sum(f, axis=2,dtype='float')/np.sum(fMask, axis=2, dtype='float')

mx = np.sum(m, axis=0)/m.shape[0]
my = np.sum(m, axis=1)/m.shape[1]
mavg = 0.5*(np.sum(mx)/mx.shape[0] + np.sum(my)/my.shape[0])

# Possibly add Gaussian filter to curves

sx = (1/mavg)*mx
sy = (1/mavg)*my

Imap = np.outer(sy,sx)
